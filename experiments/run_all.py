"""
run_all.py
==========
Run all 9 split-learning setups (3 volume levels x 3 schema levels) one by one
and write a combined summary to result_split/summary.json + summary.csv.

  python run_all.py                 # frac defaults to config.FRAC_DEFAULT (0.1)
  python run_all.py --frac 0.05     # lighter sweep
  python run_all.py --epochs 1 --frac 0.1 --max_eval 500
  python run_all.py --force         # re-run setups even if a result already exists

Each setup also writes its own result_split/<volume>__<schema>.json (via
run_experiment.run_one). By default a setup is SKIPPED if its result JSON
already exists, so re-running the script only fills in the missing setups.
Use --force to re-run everything. If a setup crashes, the error is recorded and
the sweep continues with the next setup.
"""

import argparse
import csv
import json
import os
import traceback
from datetime import datetime

from config import (VOLUME_LEVELS, SCHEMA_LEVELS, FRAC_DEFAULT, MAX_SAMPLE, EPOCHS,
                    BATCH_SIZE, LEARNING_RATE, EVAL_BATCH, RESULT_DIR, ensure_result_dir)
from run_experiment import run_one


def _existing_result(result_dir, name):
    """Return the parsed result dict for a setup if its JSON already exists, else None."""
    path = os.path.join(result_dir, f"{name}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # A corrupt/partial file shouldn't count as "done" — re-run it.
        return None


def main():
    p = argparse.ArgumentParser(description="Run all 9 split-learning setups")
    p.add_argument("--frac",   type=float, default=FRAC_DEFAULT)
    p.add_argument("--max_sample", type=int, default=MAX_SAMPLE,
                   help="Absolute cap on total TRAIN rows per setup (split across "
                        "clients). Overrides --frac when set, e.g. 4000 -> ~400/client.")
    p.add_argument("--epochs", type=int,   default=EPOCHS)
    p.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    p.add_argument("--eval_batch", type=int, default=EVAL_BATCH,
                   help=f"Eval generation batch size (default {EVAL_BATCH}).")
    p.add_argument("--lr",     type=float, default=LEARNING_RATE)
    p.add_argument("--max_eval", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-run a setup even if its result JSON already exists "
                        "(default: skip setups that are already done).")
    a = p.parse_args()

    result_dir = ensure_result_dir()
    rows = []
    sel = f"max_sample={a.max_sample}" if a.max_sample is not None else f"frac={a.frac}"
    print(f"\n{'#'*70}\n  SPLIT-LEARNING SWEEP — 9 setups "
          f"({sel}, epochs={a.epochs})\n{'#'*70}")
    if not a.force:
        print(f"  resume mode: skipping setups whose result already exists in {result_dir}")

    for vol in VOLUME_LEVELS:
        for sch in SCHEMA_LEVELS:
            name = f"{vol}__{sch}"

            # ── skip already-completed setups unless --force ──────────────────
            if not a.force:
                existing = _existing_result(result_dir, name)
                if existing is not None:
                    anls = existing.get("metrics", {}).get("anls")
                    print(f"\n  SKIP {name} — result exists "
                          f"(ANLS={anls if anls is not None else 'n/a'})")
                    rows.append({"setup": name, "volume": vol, "schema": sch,
                                 "anls": (round(anls, 6) if anls is not None else None),
                                 "status": "skipped (exists)"})
                    continue

            try:
                _, anls = run_one(vol, sch, frac=a.frac, max_sample=a.max_sample,
                                  epochs=a.epochs,
                                  batch_size=a.batch, lr=a.lr, max_eval=a.max_eval,
                                  eval_batch=a.eval_batch)
                rows.append({"setup": name, "volume": vol, "schema": sch,
                             "anls": round(anls, 6), "status": "ok"})
            except Exception as e:
                traceback.print_exc()
                rows.append({"setup": name, "volume": vol, "schema": sch,
                             "anls": None, "status": f"error: {e}"})

    # ── combined summary ──────────────────────────────────────────────────────
    summary = {"created": datetime.now().isoformat(),
               "frac": a.frac, "max_sample": a.max_sample, "epochs": a.epochs,
               "results": rows}
    with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(result_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setup", "volume", "schema", "anls", "status"])
        w.writeheader(); w.writerows(rows)

    # ── pretty 3x3 grid to stdout ─────────────────────────────────────────────
    print(f"\n{'='*70}\n  SWEEP COMPLETE — ANLS grid (rows=volume, cols=schema)\n{'='*70}")
    grid = {(r["volume"], r["schema"]): r["anls"] for r in rows}
    header = "  {:<12}".format("volume\\schema") + "".join(f"{s:>16}" for s in SCHEMA_LEVELS)
    print(header)
    for vol in VOLUME_LEVELS:
        line = "  {:<12}".format(vol)
        for sch in SCHEMA_LEVELS:
            v = grid.get((vol, sch))
            line += f"{(f'{v:.4f}' if v is not None else 'ERR'):>16}"
        print(line)
    print(f"\n  summary -> {os.path.join(result_dir, 'summary.json')}")
    print(f"  summary -> {os.path.join(result_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
