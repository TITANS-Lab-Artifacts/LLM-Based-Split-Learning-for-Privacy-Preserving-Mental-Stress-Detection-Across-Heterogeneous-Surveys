"""
sl_data.py
==========
Tokenization + collation shared by training and evaluation.

format_sample / tokenization mirror the fine-tuning scripts, including the
prompt-truncation fix: when prompt+answer exceed MAX_SEQ_LEN we trim the
PROMPT (keeping head+tail) and always keep the full answer, so long-input rows
(e.g. DASS) are never dropped.
"""

import torch

from config import (SYSTEM_PROMPT, MAX_SEQ_LEN,
                    INSTRUCTION_KEYS, INPUT_KEYS, OUTPUT_KEYS)
from data_partition import _field


def format_sample(sample, tok, for_training=True):
    user = _field(sample, INSTRUCTION_KEYS)
    inp  = _field(sample, INPUT_KEYS)
    if inp:
        user += f"\n\n{inp}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user}]
    if for_training:
        msgs.append({"role": "assistant", "content": _field(sample, OUTPUT_KEYS)})
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=not for_training)


def tokenize_train(sample, tok, max_len=MAX_SEQ_LEN):
    """Return (input_ids, labels) keeping the full answer; trim prompt if needed.
    Returns None if the sample has no answer tokens."""
    full   = tok(format_sample(sample, tok, True))["input_ids"]
    prompt = tok(format_sample(sample, tok, False))["input_ids"]
    if len(full) <= len(prompt):
        return None
    answer = full[len(prompt):]
    if len(prompt) + len(answer) <= max_len:
        ids = prompt + answer
        return ids, [-100] * len(prompt) + answer
    if len(answer) >= max_len:
        answer = answer[: max_len - 1]
        return answer, list(answer)
    keep = max_len - len(answer)
    head, tail = keep // 2, keep - keep // 2
    tp = prompt[:head] + prompt[len(prompt) - tail:]
    return tp + answer, [-100] * len(tp) + answer


def collate_train(batch, pad_id):
    """Left-pad input_ids/labels/attention_mask to the batch max length."""
    m = max(len(b[0]) for b in batch)
    ids, lbl, attn = [], [], []
    for input_ids, labels in batch:
        n = m - len(input_ids)
        ids.append([pad_id] * n + input_ids)
        lbl.append([-100] * n + labels)
        attn.append([0] * n + [1] * len(input_ids))
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(lbl, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long))


def make_batches(samples, tok, batch_size, max_len=MAX_SEQ_LEN, shuffle=True):
    """Tokenize a client's rows and yield padded (ids, labels, attn) batches.
    Skips rows with no answer tokens. Returns a list so it can be reused."""
    import random
    toks = []
    for s in samples:
        r = tokenize_train(s, tok, max_len)
        if r is not None:
            toks.append(r)
    if shuffle:
        random.shuffle(toks)
    batches = []
    for i in range(0, len(toks), batch_size):
        chunk = toks[i:i + batch_size]
        if chunk:
            batches.append(collate_train(chunk, tok.pad_token_id))
    return batches
