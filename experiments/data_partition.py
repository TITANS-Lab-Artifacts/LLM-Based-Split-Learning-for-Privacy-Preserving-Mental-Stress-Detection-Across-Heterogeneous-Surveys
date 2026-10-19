"""
data_partition.py
=================
Loads a dataset and partitions its TRAINING rows across N clients according to
the two experimental factors:

  Volume asymmetry  -> how many rows each client receives
  Schema level      -> which dataset feeds the clients (chosen in config)

Because each schema level maps to a single prepared file (homogeneous=tech,
moderate=mod_hetero, high=hetero), "schema heterogeneity" is realized by WHICH
file we load; within that file the rows are shuffled and sharded into
non-overlapping client subsets. The volume factor then decides the shard sizes.

This module is deliberately framework-free (pure Python + json) so it can be
unit-tested without torch or a GPU.
"""

import json
import os
import random

from config import (N_CLIENTS, MODERATE_VOLUME_PCT, HIGH_DOMINANT_PCT,
                    HIGH_MIN_SAMPLES, SCHEMA_FILES, DATA_DIR, RANDOM_SEED,
                    INSTRUCTION_KEYS, INPUT_KEYS, OUTPUT_KEYS)


# ── IO ──────────────────────────────────────────────────────────────────────
def _txt(v):
    return "" if v is None else (v if isinstance(v, str) else str(v))


def _field(sample, keys):
    for k in keys:
        if k in sample:
            v = _txt(sample[k]).strip()
            if v:
                return v
    return ""


def load_jsonl(path):
    """One JSON object per line; blank lines skipped."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def valid_sample(s):
    """A row is usable if it has a non-empty instruction and output."""
    return bool(_field(s, INSTRUCTION_KEYS)) and bool(_field(s, OUTPUT_KEYS))


# ── Volume vectors ───────────────────────────────────────────────────────────
def volume_counts(volume_level, total):
    """
    Return a list of N_CLIENTS integer row-counts summing to `total`,
    realizing the requested volume-asymmetry level.
    """
    n = N_CLIENTS
    if volume_level == "symmetric":
        base = total // n
        counts = [base] * n
    elif volume_level == "moderate":
        assert len(MODERATE_VOLUME_PCT) == n and sum(MODERATE_VOLUME_PCT) == 100
        counts = [total * p // 100 for p in MODERATE_VOLUME_PCT]
    elif volume_level == "high":
        dom = total * HIGH_DOMINANT_PCT // 100
        rest_total = total - dom
        # distribute the remaining 40% across the other 9 clients, each >= floor
        others = n - 1
        floor = min(HIGH_MIN_SAMPLES, rest_total // others if others else 0)
        counts = [dom] + [floor] * others
        leftover = total - sum(counts)
        # spread any leftover round-robin over the non-dominant clients
        i = 1
        while leftover > 0:
            counts[i] += 1
            leftover -= 1
            i = 1 + (i % others) if others else 0
    else:
        raise ValueError(f"unknown volume level: {volume_level}")

    # fix rounding drift so the counts sum exactly to `total`
    drift = total - sum(counts)
    counts[0] += drift            # absorb drift into the largest/first client
    counts = [max(c, 0) for c in counts]
    return counts


# ── Partitioning ──────────────────────────────────────────────────────────────
def partition_clients(volume_level, schema_level, frac=1.0, max_sample=None,
                      seed=RANDOM_SEED, data_dir=DATA_DIR, verbose=True):
    """
    Build the per-client training shards for one (volume, schema) setup.

    Returns
    -------
    clients : list[list[dict]]
        clients[i] is the list of training rows assigned to client i.
    meta : dict
        Bookkeeping (file used, totals, per-client counts).
    """
    rng = random.Random(seed)

    train_file, test_file = SCHEMA_FILES[schema_level]
    train_path = os.path.join(data_dir, train_file)
    rows = [s for s in load_jsonl(train_path) if valid_sample(s)]
    rng.shuffle(rows)

    # Select how many TRAIN rows to use (test set untouched elsewhere).
    # max_sample (absolute cap) takes priority over frac when provided.
    if max_sample is not None:
        if max_sample < N_CLIENTS:
            raise ValueError(f"max_sample must be >= N_CLIENTS ({N_CLIENTS}), got {max_sample}")
        keep = min(len(rows), max_sample)
        rows = rows[:keep]
    else:
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0,1], got {frac}")
        if frac < 1.0:
            keep = max(N_CLIENTS, int(len(rows) * frac))  # at least 1 row/client
            rows = rows[:keep]

    total = len(rows)
    counts = volume_counts(volume_level, total)

    # carve contiguous, non-overlapping shards (rows already shuffled => IID)
    clients, idx = [], 0
    for c in counts:
        clients.append(rows[idx:idx + c])
        idx += c
    # any tail rows (from flooring) go to client 0
    if idx < total:
        clients[0].extend(rows[idx:])

    meta = {
        "volume_level":  volume_level,
        "schema_level":  schema_level,
        "train_file":    train_file,
        "test_file":     test_file,
        "frac":          frac,
        "max_sample":       max_sample,
        "total_train":   total,
        "client_counts": [len(c) for c in clients],
    }
    if verbose:
        print(f"  [partition] {volume_level} x {schema_level} | file={train_file} "
              f"| {'max_sample='+str(max_sample) if max_sample is not None else 'frac='+str(frac)} "
              f"| total={total}")
        print(f"  [partition] client rows: {meta['client_counts']} "
              f"(sum={sum(meta['client_counts'])})")
    return clients, meta


def load_test(schema_level, data_dir=DATA_DIR):
    """Load the FULL test set for a schema level (never subsampled)."""
    _, test_file = SCHEMA_FILES[schema_level]
    rows = [s for s in load_jsonl(os.path.join(data_dir, test_file)) if valid_sample(s)]
    return rows, test_file
