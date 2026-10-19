"""
split_text_datasets.py
----------------------
Performs train/test splitting and dataset mixing for the four mental
health JSONL files.

Tasks:
  1. Per-file split     — 500 test samples from each file (PER_FILE_N)
                          → test_{name}.jsonl  +  train_{name}.jsonl

  2. Moderate heterogeneous mix (tech + student)
                        — merge mh_tech + mh_student, 500 test samples (MIX_MOD_N)
                          → test_mod_hetero.jsonl  +  train_mod_hetero.jsonl

  3. Full heterogeneous mix (all four datasets)
                        — merge all four, 500 test samples (MIX_FULL_N)
                          → test_hetero.jsonl  +  train_hetero.jsonl

Input: data/processed/jsonl/ (written by the four convert_*.py scripts).
All output files are saved under OUTPUT_DIR (data/processed/text/).
Splits are reproducible via RANDOM_SEED.
"""

import json
import os
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT_DIR, "data", "processed", "jsonl")

FILES = {
    "mh_dass_dataset":    os.path.join(BASE_DIR, "mh_dass_dataset.jsonl"),
    "mh_general_dataset": os.path.join(BASE_DIR, "mh_general_dataset.jsonl"),
    "mh_student_dataset": os.path.join(BASE_DIR, "mh_student_dataset.jsonl"),
    "mh_tech_dataset":    os.path.join(BASE_DIR, "mh_tech_dataset.jsonl"),
}

OUTPUT_DIR   = os.path.join(ROOT_DIR, "data", "processed", "text")
PER_FILE_N   = 500   # test samples per individual file
MIX_MOD_N    = 500   # test samples for moderate heterogeneous mix
MIX_FULL_N   = 500   # test samples for full heterogeneous mix
RANDOM_SEED  = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load all lines from a JSONL file into a list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples, path):
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def split(samples, n_test):
    """
    Randomly draw n_test samples for test set.
    Returns (train_samples, test_samples).
    Raises ValueError if dataset is too small.
    """
    if len(samples) < n_test:
        raise ValueError(
            f"Dataset has only {len(samples)} samples but {n_test} were "
            f"requested for the test set."
        )
    shuffled = samples.copy()
    random.shuffle(shuffled)
    return shuffled[n_test:], shuffled[:n_test]   # train, test


def report(tag, train, test, out_train, out_test):
    """Print a summary line and save both splits."""
    save_jsonl(train, out_train)
    save_jsonl(test,  out_test)
    total = len(train) + len(test)
    print(f"  {tag}")
    print(f"    total={total:>10,}  train={len(train):>10,}  test={len(test):>6,}")
    print(f"    train → {os.path.basename(out_train)}")
    print(f"    test  → {os.path.basename(out_test)}")


# ── Load all files ────────────────────────────────────────────────────────────

print("Loading files...")
data = {}
for name, path in FILES.items():
    data[name] = load_jsonl(path)
    print(f"  {name}: {len(data[name]):,} samples")


# ── Task 1: per-file splits ───────────────────────────────────────────────────

print("\n── Task 1: Individual file splits (1,000 test each) ──────────────────")
per_file_trains = {}   # keep train splits for reuse in mixing tasks

for name, samples in data.items():
    train, test = split(samples, PER_FILE_N)
    per_file_trains[name] = train

    out_train = os.path.join(OUTPUT_DIR, f"train_{name}.jsonl")
    out_test  = os.path.join(OUTPUT_DIR, f"test_{name}.jsonl")
    report(name, train, test, out_train, out_test)


# ── Task 2: moderate heterogeneous mix (tech + student) ──────────────────────

print("\n── Task 2: Moderate heterogeneous mix — tech + student (2,000 test) ──")

mod_mix = (
    data["mh_tech_dataset"] +
    data["mh_student_dataset"]
)
random.shuffle(mod_mix)

train_mod, test_mod = split(mod_mix, MIX_MOD_N)

report(
    "mh_tech + mh_student",
    train_mod, test_mod,
    os.path.join(OUTPUT_DIR, "train_mod_hetero.jsonl"),
    os.path.join(OUTPUT_DIR, "test_mod_hetero.jsonl"),
)


# ── Task 3: full heterogeneous mix (all four) ─────────────────────────────────

print("\n── Task 3: Full heterogeneous mix — all four (2,000 test) ────────────")

full_mix = (
    data["mh_dass_dataset"] +
    data["mh_general_dataset"] +
    data["mh_student_dataset"] +
    data["mh_tech_dataset"]
)
random.shuffle(full_mix)

train_full, test_full = split(full_mix, MIX_FULL_N)

report(
    "all four datasets",
    train_full, test_full,
    os.path.join(OUTPUT_DIR, "train_hetero.jsonl"),
    os.path.join(OUTPUT_DIR, "test_hetero.jsonl"),
)


# ── Final summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("OUTPUT FILES")
print("=" * 62)
files_out = sorted(os.listdir(OUTPUT_DIR))
total_size = 0
for fname in files_out:
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    total_size += os.path.getsize(fpath)
    # quick line count
    with open(fpath, "r", encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip())
    print(f"  {fname:<40}  {n_lines:>10,} samples  {size_mb:>7.1f} MB")
print("-" * 62)
print(f"  {'Total':<40}  {'':>10}  {total_size/(1024*1024):>7.1f} MB")
print(f"\nAll files saved to: {OUTPUT_DIR}")
print("=" * 62)