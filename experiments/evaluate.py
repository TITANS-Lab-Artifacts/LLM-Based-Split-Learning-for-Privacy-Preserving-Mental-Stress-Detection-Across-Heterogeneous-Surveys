"""
evaluate.py
===========
Evaluate the trained split model (client + server) on a test set and report ANLS.

Generation through the split pipeline is autoregressive and BATCHED: a batch of
prompts is left-padded to a common length, then at each step the client embeds
the whole batch into cut activations, the server produces next-token logits, we
greedily pick one token per sequence, append it, and repeat until every sequence
hits EOS / <|eot_id|> or MAX_NEW_TOKENS. Left padding keeps every sequence's
"current" position at the right edge, so a single logits[:, -1] slice gives the
next-token logits for all rows.

DIAGNOSTIC: when config.DIAGNOSTIC_GRAD_STATS is on, the first eval batch prints
the RAW generated token ids (before skip_special_tokens) for the first two rows.
If token 0 is EOS / <|eot_id|>, the model is emitting end-of-turn immediately —
the direct cause of empty predictions and ANLS 0.

Batch size is config.EVAL_BATCH (adjustable).
"""

import torch

from config import (MAX_NEW_TOKENS, MAX_SEQ_LEN, EVAL_BATCH, OUTPUT_KEYS,
                    DIAGNOSTIC_GRAD_STATS)
from sl_data import format_sample
from anls import compute_anls
from data_partition import _field
from progress import ProgressBar


def _eos_ids(tok):
    ids = {tok.eos_token_id}
    try:
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot >= 0:
            ids.add(eot)
    except Exception:
        pass
    return ids


def _prep_prompt_ids(s, tok, max_new_tokens):
    """Tokenize one prompt; trim from the left if it would leave no room to gen."""
    ids = tok(format_sample(s, tok, for_training=False))["input_ids"]
    cap = MAX_SEQ_LEN - max_new_tokens
    if len(ids) > cap:
        ids = ids[-cap:]
    return ids


@torch.no_grad()
def generate_batch(client, server, tok, prompts, server_device, client_device,
                   max_new_tokens=MAX_NEW_TOKENS, eos_ids=None, debug=False):
    """
    Greedy-decode a BATCH of prompts (list of token-id lists) through client->server.
    Returns a list of generated-token-id lists (one per prompt), EOS excluded.

    Left-pads the batch; tracks per-sequence completion; stops when all done.
    """
    eos_ids = eos_ids or _eos_ids(tok)
    pad_id = tok.pad_token_id
    B = len(prompts)

    # ── left-pad prompts to a common length ──────────────────────────────────
    maxlen = max(len(p) for p in prompts)
    seqs, attn = [], []
    for p in prompts:
        n = maxlen - len(p)
        seqs.append([pad_id] * n + list(p))
        attn.append([0] * n + [1] * len(p))
    seqs = torch.tensor(seqs, dtype=torch.long)        # [B, T]
    attn = torch.tensor(attn, dtype=torch.long)        # [B, T]

    generated = [[] for _ in range(B)]
    finished  = [False] * B
    raw_first_tokens = [[] for _ in range(B)]          # for the debug print

    for _step in range(max_new_tokens):
        cut = client(seqs)                              # [B,T,H] client device
        _, logits = server(cut.detach().to(server_device),
                           attn.to(server_device), labels=None)
        next_ids = logits[:, -1, :].argmax(dim=-1).to("cpu")   # [B]

        new_col_tok  = torch.full((B, 1), pad_id, dtype=torch.long)
        new_col_attn = torch.zeros((B, 1), dtype=torch.long)
        for b in range(B):
            if finished[b]:
                continue                                # keep padding finished rows
            tid = int(next_ids[b].item())
            if debug and len(raw_first_tokens[b]) < 10:
                raw_first_tokens[b].append(tid)
            if tid in eos_ids:
                finished[b] = True
                continue
            generated[b].append(tid)
            new_col_tok[b, 0]  = tid
            new_col_attn[b, 0] = 1

        if all(finished):
            break
        seqs = torch.cat([seqs, new_col_tok], dim=1)
        attn = torch.cat([attn, new_col_attn], dim=1)
        if seqs.shape[1] >= MAX_SEQ_LEN:
            break

    if debug:
        eos_set = sorted(eos_ids)
        for b in range(min(B, 2)):
            toks = raw_first_tokens[b]
            decoded = [tok.decode([t]) for t in toks]
            print(f"\n  [diag] eval row {b}: first raw token ids = {toks}")
            print(f"  [diag] eval row {b}: decoded            = {decoded}")
            if toks and toks[0] in eos_ids:
                print(f"  [diag] eval row {b}: token 0 is EOS/<|eot_id|> "
                      f"(eos ids={eos_set}) -> empty prediction. Model is "
                      f"emitting end-of-turn immediately.")

    return generated


@torch.no_grad()
def generate_one(client, server, tok, prompt_ids, server_device, client_device,
                 max_new_tokens=MAX_NEW_TOKENS, eos_ids=None):
    """Single-prompt convenience wrapper around generate_batch (kept for compat)."""
    return generate_batch(client, server, tok, [list(prompt_ids)],
                          server_device, client_device, max_new_tokens, eos_ids)[0]


@torch.no_grad()
def evaluate_split(client, server, tok, test_rows, server_device, client_device,
                   max_new_tokens=MAX_NEW_TOKENS, batch_size=EVAL_BATCH,
                   log_every=100, max_eval=None):
    """
    Run the split model over test_rows in batches; return (mean_anls, results).
    max_eval optionally caps the number of test rows scored (None = all).
    """
    server.hf().eval()
    client.eval()
    eos_ids = _eos_ids(tok)

    rows = test_rows if max_eval is None else test_rows[:max_eval]
    scores, results = [], []
    eval_bar = ProgressBar(len(rows), prefix="eval ")

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [_prep_prompt_ids(s, tok, max_new_tokens) for s in batch]
        debug = DIAGNOSTIC_GRAD_STATS and start == 0      # only the first batch
        gens = generate_batch(client, server, tok, prompts,
                              server_device, client_device, max_new_tokens,
                              eos_ids, debug=debug)
        for j, (s, gen_ids) in enumerate(zip(batch, gens)):
            pred = tok.decode(gen_ids, skip_special_tokens=True).strip()
            gt = _field(s, OUTPUT_KEYS)
            sc, pc, gc = compute_anls(pred, gt)
            scores.append(sc)
            results.append({"index": start + j + 1,
                            "instruction": s.get("instruction", ""),
                            "expected": gt, "generated": pred,
                            "pred_core": pc, "gt_core": gc, "anls": round(sc, 4)})
        eval_bar.update(step=len(batch), extra=f"ANLS={sum(scores)/len(scores):.4f}")
    eval_bar.close()
    mean_anls = sum(scores) / max(len(scores), 1)
    return mean_anls, results
