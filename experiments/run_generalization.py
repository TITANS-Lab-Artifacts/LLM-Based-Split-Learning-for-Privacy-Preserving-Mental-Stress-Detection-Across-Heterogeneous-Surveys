"""
run_generalization.py
=====================
Train ONE split-learning model on the symmetric x high setup, then evaluate that
single trained model against the six test_*.jsonl files found in the data dir and
report every ANLS value to a CSV named result_split_generalization.csv.

Pipeline
--------
  1. Partition the symmetric x high TRAIN data across N_CLIENTS.
  2. Build client + server and split-train once.
  3. Discover all test_*.jsonl files in DATA_DIR.
  4. Evaluate the SAME trained model on each test file -> mean ANLS.
  5. Write result_split/result_split_generalization.csv.

CLI
---
  python run_generalization.py
  python run_generalization.py --frac 0.1 --epochs 3 --max_eval 500
  python run_generalization.py --max_sample 2000 --eval_batch 64
  # Restrict / override the test files explicitly (otherwise auto-discovered):
  python run_generalization.py --test_files test_hetero.jsonl test_mod_hetero.jsonl
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime

import torch

from config import (FRAC_DEFAULT, MAX_SAMPLE, EPOCHS, BATCH_SIZE, LEARNING_RATE,
                    MAX_SEQ_LEN, MAX_NEW_TOKENS, EVAL_BATCH, LORA_R, LORA_ALPHA,
                    PROJ_HIDDEN, N_CLIENTS, MODEL_ID, RANDOM_SEED, DATA_DIR,
                    ensure_result_dir, get_devices)
from data_partition import partition_clients, load_jsonl, valid_sample

# Fixed training setup for this generalization study.
TRAIN_VOLUME = "symmetric"
TRAIN_SCHEMA = "high"


def discover_test_files(data_dir, explicit=None):
    """Return a sorted list of test_*.jsonl basenames in data_dir.
    If `explicit` is given, validate those instead of globbing."""
    if explicit:
        files = []
        for f in explicit:
            base = os.path.basename(f)
            if not os.path.isfile(os.path.join(data_dir, base)):
                raise FileNotFoundError(f"test file not found in {data_dir}: {base}")
            files.append(base)
        return files
    paths = glob.glob(os.path.join(data_dir, "test_*.jsonl"))
    return sorted(os.path.basename(p) for p in paths)


def load_test_file(test_file, data_dir=DATA_DIR):
    """Load and filter one test file by name (independent of SCHEMA_FILES)."""
    rows = [s for s in load_jsonl(os.path.join(data_dir, test_file)) if valid_sample(s)]
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Train symmetric x high once, evaluate on six test sets")
    p.add_argument("--frac", type=float, default=FRAC_DEFAULT,
                   help="Fraction of TRAIN data to use (0<frac<=1). Test is full.")
    p.add_argument("--max_sample", type=int, default=MAX_SAMPLE,
                   help="Absolute cap on total TRAIN rows (overrides --frac when set).")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--eval_batch", type=int, default=EVAL_BATCH,
                   help=f"Eval generation batch size (default {EVAL_BATCH}).")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--max_eval", type=int, default=None,
                   help="Cap number of test rows scored per file (default: all).")
    p.add_argument("--test_files", nargs="*", default=None,
                   help="Explicit test file names (default: auto-discover test_*.jsonl).")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = p.parse_args()

    import random
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    server_device, client_device, n_gpu = get_devices()
    result_dir = ensure_result_dir()

    sel = f"max_sample={a.max_sample}" if a.max_sample is not None else f"frac={a.frac}"
    print(f"\n{'#'*70}")
    print(f"  GENERALIZATION RUN — train {TRAIN_VOLUME} x {TRAIN_SCHEMA} "
          f"({sel}, epochs={a.epochs})")
    print(f"  devices: server={server_device} client={client_device} (GPUs={n_gpu})")
    print(f"{'#'*70}")

    # ── discover the six test files ───────────────────────────────────────────
    test_files = discover_test_files(DATA_DIR, explicit=a.test_files)
    if not test_files:
        raise SystemExit(f"No test_*.jsonl files found in {DATA_DIR}")
    print(f"  test files ({len(test_files)}): {test_files}")
    if len(test_files) != 6:
        print(f"  NOTE: expected 6 test files, found {len(test_files)}; "
              f"proceeding with all discovered files.")

    # ── 1. Partition the symmetric x high training data ───────────────────────
    clients_rows, meta = partition_clients(TRAIN_VOLUME, TRAIN_SCHEMA, frac=a.frac,
                                           max_sample=a.max_sample, seed=a.seed)

    # ── 2. Build models + train once ──────────────────────────────────────────
    from model_server import ServerModel, load_tokenizer
    from model_client import ClientModel
    from split_train import train_split
    from evaluate import evaluate_split

    tok = load_tokenizer()
    print("  building server (LoRA LLaMA trunk) ...")
    server = ServerModel(server_device)
    print("  building client (LoRA embedding + projection head) ...")
    client = ClientModel(
        embed_tokens=server.get_embed_tokens(),
        hidden_size=server.hidden_size, vocab_size=server.vocab_size,
        lora_r=LORA_R, lora_alpha=LORA_ALPHA, bottleneck=PROJ_HIDDEN,
        device=client_device)
    print(f"  client trainable params: {client.param_count():,}")
    print(f"  server trainable params: "
          f"{sum(p.numel() for p in server.trainable_parameters()):,}")

    print(f"\n  training on {TRAIN_VOLUME} x {TRAIN_SCHEMA} ...")
    train_stats = train_split(
        client, server, tok, clients_rows, server_device, client_device,
        epochs=a.epochs, batch_size=a.batch, lr=a.lr, max_len=MAX_SEQ_LEN)

    # ── 3 & 4. Evaluate the SAME trained model on each test file ──────────────
    rows = []
    for test_file in test_files:
        test_rows = load_test_file(test_file)
        print(f"\n  evaluating on {test_file} ({len(test_rows)} rows, "
              f"batch={a.eval_batch}) ...")
        if not test_rows:
            print(f"  WARNING: {test_file} has no valid rows; recording ANLS=None.")
            rows.append({"train_setup": f"{TRAIN_VOLUME}__{TRAIN_SCHEMA}",
                         "test_file": test_file, "test_rows": 0,
                         "rows_scored": 0, "anls": None, "status": "empty"})
            continue
        try:
            mean_anls, _ = evaluate_split(
                client, server, tok, test_rows, server_device, client_device,
                max_new_tokens=MAX_NEW_TOKENS, batch_size=a.eval_batch,
                max_eval=a.max_eval)
            scored = len(test_rows) if a.max_eval is None else min(len(test_rows), a.max_eval)
            print(f"  >>> {test_file}  ANLS = {mean_anls:.4f} ({mean_anls*100:.2f}%)")
            rows.append({"train_setup": f"{TRAIN_VOLUME}__{TRAIN_SCHEMA}",
                         "test_file": test_file, "test_rows": len(test_rows),
                         "rows_scored": scored, "anls": round(mean_anls, 6),
                         "status": "ok"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({"train_setup": f"{TRAIN_VOLUME}__{TRAIN_SCHEMA}",
                         "test_file": test_file, "test_rows": len(test_rows),
                         "rows_scored": 0, "anls": None, "status": f"error: {e}"})

    # ── 5. Write the CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(result_dir, "result_split_generalization.csv")
    fieldnames = ["train_setup", "test_file", "test_rows", "rows_scored", "anls", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # also drop a small JSON sidecar with run metadata
    json_path = os.path.join(result_dir, "result_split_generalization.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"created": datetime.now().isoformat(),
                   "train_setup": f"{TRAIN_VOLUME}__{TRAIN_SCHEMA}",
                   "model": MODEL_ID, "n_clients": N_CLIENTS,
                   "frac": a.frac, "max_sample": a.max_sample, "epochs": a.epochs,
                   "batch_size": a.batch, "lr": a.lr, "eval_batch": a.eval_batch,
                   "max_eval": a.max_eval, "train_partition": meta,
                   "train_stats": train_stats, "results": rows},
                  f, indent=2, ensure_ascii=False)

    # ── pretty print ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  GENERALIZATION COMPLETE — trained on {TRAIN_VOLUME} x {TRAIN_SCHEMA}")
    print(f"{'='*70}")
    print("  {:<34}{:>12}".format("test_file", "ANLS"))
    for r in rows:
        v = r["anls"]
        print("  {:<34}{:>12}".format(
            r["test_file"], (f"{v:.4f}" if v is not None else r["status"])))
    print(f"\n  csv  -> {csv_path}")
    print(f"  json -> {json_path}")

    # ── free GPU memory ───────────────────────────────────────────────────────
    del server, client
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
