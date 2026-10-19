"""
run_ft_transformer.py
---------------------
Automatically runs ft_transformer.py on every train/test CSV pair found in a
directory (or on an explicit hard-coded list), one after another, and writes a
summary table at the end.

USAGE:
  # Auto-discover all train_*.csv / test_*.csv pairs in the current folder:
  python run_ft_transformer.py

  # Point at a specific data directory:
  python run_ft_transformer.py --data-dir /path/to/csvs

  # Override epochs / batch size for every run:
  python run_ft_transformer.py --epochs 30 --batch 256

  # Force a target column for datasets that have it (skipped if absent):
  python run_ft_transformer.py --target treatment

Pairs are matched by the suffix after "train_" / "test_". For example
train_student.csv pairs with test_student.csv (key = "student").

Per-dataset target overrides can be set in TARGET_OVERRIDES below; otherwise
ft_transformer.py auto-detects the target.
"""

import argparse
import glob
import json
import os
import sys
import time

# Import the corrected training module (must sit next to this file).
import ft_transformer as ft


# Optional per-dataset target overrides. Key = the suffix after train_/test_.
# Leave empty to let the model auto-detect for every dataset.
TARGET_OVERRIDES = {
    # "student":     "Depression",
    # "general":     "treatment",
    # "hetero":      "treatment",
    # "mod_hetero":  "Depression",
    # "dass":        "depression_level",
    # "osmi":        "sought_treatment",
}


def discover_pairs(data_dir):
    """Find every (train, test) CSV pair by matching the suffix after the prefix."""
    trains, tests = {}, {}
    for path in glob.glob(os.path.join(data_dir, "train_*.csv")):
        key = os.path.basename(path)[len("train_"):-len(".csv")]
        trains[key] = path
    for path in glob.glob(os.path.join(data_dir, "test_*.csv")):
        key = os.path.basename(path)[len("test_"):-len(".csv")]
        tests[key] = path

    pairs = []
    for key in sorted(trains):
        if key in tests:
            pairs.append((key, trains[key], tests[key]))
        else:
            print(f"  [WARN] train_{key}.csv has no matching test_{key}.csv — skipping.")
    for key in sorted(tests):
        if key not in trains:
            print(f"  [WARN] test_{key}.csv has no matching train_{key}.csv — skipping.")
    return pairs


def main():
    p = argparse.ArgumentParser(description="Batch-run FT-Transformer over all pairs")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed", "tabular"),
                   help="Folder containing the CSVs (default: data/processed/tabular)")
    p.add_argument("--epochs", type=int, default=ft.EPOCHS)
    p.add_argument("--batch",  type=int, default=ft.BATCH_SIZE)
    p.add_argument("--lr",     type=float, default=ft.LEARNING_RATE)
    p.add_argument("--d_token", type=int, default=ft.D_TOKEN)
    p.add_argument("--layers", type=int, default=ft.N_LAYERS)
    p.add_argument("--target", default=None,
                   help="Force this target for ALL datasets that contain it "
                        "(overrides TARGET_OVERRIDES). Datasets lacking it auto-detect.")
    p.add_argument("--only", nargs="*", default=None,
                   help="Run only these dataset keys (e.g. --only student dass)")
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    print("=" * 60)
    print("  BATCH FT-TRANSFORMER RUNNER")
    print(f"  Data dir : {data_dir}")
    print(f"  Device   : {ft.DEVICE.upper()}")
    print("=" * 60)

    pairs = discover_pairs(data_dir)
    if args.only:
        pairs = [t for t in pairs if t[0] in set(args.only)]
    if not pairs:
        sys.exit("  No train/test pairs found. Check --data-dir.")

    print(f"\n  Found {len(pairs)} dataset pair(s): {[k for k, _, _ in pairs]}\n")

    summary = []
    for i, (key, train_path, test_path) in enumerate(pairs, 1):
        # Resolve target: CLI --target (if the dataset has it) > override > auto
        target = None
        if args.target:
            import pandas as pd
            cols = pd.read_csv(train_path, nrows=1).columns
            target = args.target if args.target in cols else None
        if target is None:
            target = TARGET_OVERRIDES.get(key)

        print("\n" + "#" * 60)
        print(f"#  [{i}/{len(pairs)}]  DATASET: {key}"
              f"   (target={target or 'auto'})")
        print("#" * 60)

        t0 = time.time()
        try:
            results = ft.run(
                train_path=train_path, test_path=test_path, target_col=target,
                epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                d_token=args.d_token, n_layers=args.layers,
            )
            elapsed = time.time() - t0
            row = {
                "dataset": key,
                "task_type": results.get("task_type"),
                "target": results["run_info"]["target_column"],
                "accuracy": results.get("accuracy"),
                "f1_macro": results.get("f1_macro"),
                "roc_auc": results.get("roc_auc"),
                "rmse": results.get("rmse"),
                "r2": results.get("r2"),
                "test_rows": results["run_info"]["test_rows"],
                "n_features": results["run_info"]["n_features"],
                "seconds": round(elapsed, 1),
                "status": "ok",
            }
        except SystemExit as e:
            row = {"dataset": key, "status": f"skipped: {e}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            row = {"dataset": key, "status": f"error: {type(e).__name__}: {e}"}
        summary.append(row)

    # ---- summary table --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    header = f"{'dataset':<14}{'task':<12}{'target':<18}{'acc':>7}{'f1m':>7}{'auc':>7}{'sec':>7}"
    print(header)
    print("-" * len(header))
    for r in summary:
        if r.get("status") != "ok":
            print(f"{r['dataset']:<14}{r.get('status', '')}")
            continue
        acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else "-"
        f1m = f"{r['f1_macro']:.3f}" if r.get("f1_macro") is not None else "-"
        auc = f"{r['roc_auc']:.3f}" if r.get("roc_auc") is not None else (
            f"r2={r['r2']:.3f}" if r.get("r2") is not None else "-")
        print(f"{r['dataset']:<14}{str(r['task_type']):<12}"
              f"{str(r['target'])[:17]:<18}{acc:>7}{f1m:>7}{auc:>7}{r['seconds']:>7}")

    out = os.path.join(data_dir, "results", "batch_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Full summary saved -> {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
