"""
split_train.py
==============
The split-learning training loop for ONE setup (one volume x schema cell).

Faithful SL semantics across two GPUs:
  * Client (cuda:1): token ids -> cut activation.
  * Cut: the activation is detached, moved to the server GPU, and marked
    requires_grad so the server can compute d(loss)/d(cut).
  * Server (cuda:0): cut activation -> LLaMA trunk -> LM loss; backward gives
    the gradient at the cut.
  * That cut-gradient is shipped back to the client GPU and used to backprop
    through the client (projection head + embedding LoRA).
Activations and gradients are exchanged PER BATCH, which is exactly the
information that crosses the client/server boundary in split learning.

Federated aspect: ONE shared server trunk and ONE shared client front-end.
Within a global epoch we iterate clients in order; for each client we run its
local batches. Client order is shuffled each epoch.

DIAGNOSTICS (config.DIAGNOSTIC_GRAD_STATS): once per epoch we print the cut
activation stats and the transported cut-gradient magnitude, so a "loss stuck
at ~4.4" failure can be diagnosed as either (a) tiny/dead cut gradient
(plumbing) or (b) healthy gradient but the trunk still can't fit (architecture
/ budget).
"""

import time
import torch

from config import (LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM, WARMUP_RATIO,
                    BATCH_SIZE, EPOCHS, MAX_SEQ_LEN, LORA_R, LORA_ALPHA,
                    PROJ_HIDDEN, DIAGNOSTIC_GRAD_STATS, DIAGNOSTIC_IDENTITY)
from sl_data import make_batches


def _build_scheduler(optimizer, total_steps):
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), max(total_steps, 1))


def train_split(client, server, tok, clients_rows, server_device, client_device,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                max_len=MAX_SEQ_LEN, log_every=20):
    """
    Train client+server on the partitioned client rows.

    Returns a dict with the loss history (for the results file).
    """
    import random
    from progress import ProgressBar

    # ── Pre-tokenize every client's rows into reusable padded batches ─────────
    print(f"  [train] tokenizing {len(clients_rows)} client shards ...")
    tok_bar = ProgressBar(len(clients_rows), prefix="tokenize")
    client_batches = []
    total_batches = 0
    for ci, rows in enumerate(clients_rows):
        b = make_batches(rows, tok, batch_size, max_len, shuffle=True)
        client_batches.append(b)
        total_batches += len(b)
        tok_bar.update(extra=f"client {ci}: {len(b)} batches")
    tok_bar.close(extra=f"{total_batches} batches total")
    print(f"  [train] {len(clients_rows)} clients, "
          f"{total_batches} batches/epoch, {epochs} epoch(s)")
    if total_batches == 0:
        return {"loss_history": [], "note": "no usable batches"}

    if DIAGNOSTIC_IDENTITY:
        print("  [train] DIAGNOSTIC_IDENTITY=True — cut is the raw frozen "
              "embedding; only the server trunk LoRA is trained.")

    # ── One optimizer over BOTH sides' trainable params ───────────────────────
    # In identity mode the client has no trainable params; only the server.
    params = server.trainable_parameters() + client.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = total_batches * epochs
    scheduler = _build_scheduler(optimizer, total_steps)
    # NOTE: no GradScaler. The cut-layer gradient is read off cut_server.grad and
    # transported to the client; a scaler would scale that gradient on the server
    # side while the client's resulting grads would never be unscaled. The LoRA
    # params are fp32 (only the frozen base trunk is fp16), so plain fp16 training
    # is stable here.

    server.hf().train()
    client.train()

    loss_history = []
    step = 0
    t0 = time.time()
    train_bar = ProgressBar(total_steps, prefix="train")

    for epoch in range(1, epochs + 1):
        order = list(range(len(clients_rows)))
        random.shuffle(order)                      # shuffle client visitation
        ep_loss, ep_n = 0.0, 0
        # per-epoch diagnostic accumulators (first batch of the epoch only)
        diag_done = False

        for ci in order:
            batches = client_batches[ci]
            for (ids, labels, attn) in batches:
                optimizer.zero_grad(set_to_none=True)

                # ---- CLIENT forward (cuda:1): ids -> cut activation (fp32) ----
                cut_client = client(ids)            # [B,T,H] on client_device

                # ---- CUT: detach, cross to server GPU, require grad ----
                cut_server = cut_client.detach().to(server_device).requires_grad_(True)

                # ---- SERVER forward+backward (cuda:0) ----
                # server.forward casts the cut to the model dtype (fp16) internally.
                loss, _ = server(cut_server, attn.to(server_device),
                                 labels.to(server_device))
                loss.backward()                     # fills cut_server.grad + server grads

                # ---- transport cut-gradient back to the client GPU ----
                cut_grad = cut_server.grad.detach().to(client_device, dtype=cut_client.dtype)

                # ---- diagnostics: is signal actually flowing across the cut? --
                if DIAGNOSTIC_GRAD_STATS and not diag_done:
                    with torch.no_grad():
                        cg = cut_grad.float()
                        ca = cut_client.float()
                        print(f"\n  [diag] epoch {epoch} c{ci}: "
                              f"cut act mean={ca.mean():.4e} std={ca.std():.4e} | "
                              f"cut grad abs-mean={cg.abs().mean():.4e} "
                              f"max={cg.abs().max():.4e} | loss={loss.item():.4f}")
                    diag_done = True

                # ---- CLIENT backward using the transported gradient ----
                # In identity mode cut_client has no trainable upstream params,
                # so this backward is a harmless no-op; skip it to save time.
                if client.trainable_parameters():
                    cut_client.backward(cut_grad)

                # ---- optimizer step (both sides) ----
                if params:
                    torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()

                lv = loss.item()
                ep_loss += lv; ep_n += 1; step += 1
                train_bar.update(extra=(f"E{epoch}/{epochs} c{ci} "
                                        f"loss={lv:.4f} avg={ep_loss/ep_n:.4f}"))

        avg = ep_loss / max(ep_n, 1)
        loss_history.append(avg)
        print(f"\n  [train] epoch {epoch}/{epochs} done — avg loss {avg:.4f} "
              f"({(time.time()-t0)/60:.1f} min elapsed)")

    train_bar.close()
    print(f"  [train] complete — {(time.time()-t0)/60:.1f} min, {step} steps")
    return {"loss_history": loss_history, "steps": step,
            "minutes": round((time.time() - t0) / 60, 2)}
