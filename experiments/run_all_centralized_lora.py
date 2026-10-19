"""
run_all.py
----------
Batch-runs both fine-tuning scripts (finetune_llama.py and finetune_openbio.py)
over every train/test dataset pair, one at a time.

Each train file is paired with the test file of the same suffix, e.g.
  train_hetero.jsonl            -> test_hetero.jsonl
  train_mh_dass_dataset.jsonl   -> test_mh_dass_dataset.jsonl

For each pair, both models are run sequentially. Training samples are capped
(N_TRAIN below). Per-run stdout/stderr is logged to ./run_logs/.

USAGE:
  python run_all.py                 # run everything
  python run_all.py --dry-run       # just print the commands
  python run_all.py --models llama  # only the 3B model (choices: llama, openbio)
  python run_all.py --n_train 2000  # override the training-sample cap

NOTE:
  Both fine-tune scripts write results to results/<train>__<test>/results.json.
  Because both models share that folder, this runner renames each result file
  to results.<model>.json after the run so the second model does not overwrite
  the first.
"""

import argparse, os, shutil, subprocess, sys, time
from datetime import datetime

# ── EDIT THESE IF NEEDED ──────────────────────────────────────────────────────
N_TRAIN = 2000                      # cap on training samples per run (adjustable)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "processed", "text")   # folder containing the .jsonl files

# train file -> test file (same-suffix pairing)
PAIRS = [
    ("train_hetero.jsonl",                "test_hetero.jsonl"),
    ("train_mod_hetero.jsonl",            "test_mod_hetero.jsonl"),
    ("train_mh_dass_dataset.jsonl",       "test_mh_dass_dataset.jsonl"),
    ("train_mh_general_dataset.jsonl",    "test_mh_general_dataset.jsonl"),
    ("train_mh_student_dataset.jsonl",    "test_mh_student_dataset.jsonl"),
    ("train_mh_tech_dataset.jsonl",       "test_mh_tech_dataset.jsonl"),
]

# model key -> fine-tune script
MODELS = {
    "llama":   "finetune_llama.py",
    "openbio": "finetune_openbio.py",
}
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR = "run_logs"


def result_dir(train_file, test_file):
    tr = os.path.splitext(os.path.basename(train_file))[0]
    te = os.path.splitext(os.path.basename(test_file))[0]
    return os.path.join("results", f"{tr}__{te}")


def main():
    ap = argparse.ArgumentParser(description="Batch-run fine-tuning over all dataset pairs.")
    ap.add_argument("--n_train", type=int, default=N_TRAIN,
                    help=f"Cap on training samples per run (default {N_TRAIN}).")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                    help="Which model(s) to run (default: both).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running them.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a run if its renamed result file already exists.")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going if a run fails (default: stop).")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    # Build the job list: every (model, pair) combination.
    jobs = [(m, tr, te) for tr, te in PAIRS for m in args.models]
    total = len(jobs)
    print(f"\n{'='*70}")
    print(f"  BATCH RUN — {len(args.models)} model(s) x {len(PAIRS)} pair(s) = {total} runs")
    print(f"  n_train cap : {args.n_train}")
    print(f"  models      : {', '.join(args.models)}")
    print(f"{'='*70}\n")

    # Sanity-check that all referenced files exist (warn only).
    missing = []
    for tr, te in PAIRS:
        for fn in (tr, te):
            if not os.path.isfile(os.path.join(DATA_DIR, fn)):
                missing.append(fn)
    for m in args.models:
        if not os.path.isfile(MODELS[m]):
            missing.append(MODELS[m])
    if missing:
        print("  WARNING — these files were not found in the current directory:")
        for fn in sorted(set(missing)):
            print(f"    - {fn}")
        print()

    summary = []
    t_start = time.time()

    for idx, (model, train_file, test_file) in enumerate(jobs, 1):
        script = MODELS[model]
        rdir = result_dir(train_file, test_file)
        final_result = os.path.join(rdir, f"results.{model}.json")

        tag = f"[{idx}/{total}] {model:8s}  {train_file}  ->  {test_file}"
        print(f"\n{'-'*70}\n  {tag}\n{'-'*70}")

        if args.skip_existing and os.path.isfile(final_result):
            print(f"  SKIP — {final_result} already exists.")
            summary.append((tag, "skipped", 0.0))
            continue

        cmd = [
            sys.executable, script,
            "--train", os.path.join(DATA_DIR, train_file),
            "--test",  os.path.join(DATA_DIR, test_file),
            "--n_train", str(args.n_train),
        ]
        print("  CMD:", " ".join(cmd))

        if args.dry_run:
            summary.append((tag, "dry-run", 0.0))
            continue

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"{model}__{os.path.splitext(train_file)[0]}__{stamp}.log")
        print(f"  LOG: {log_path}")

        # Force the child process to emit UTF-8 so box-drawing chars don't trip
        # the Windows cp1252 codec, and decode the pipe as UTF-8 on our side.
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        t0 = time.time()
        rc = 0
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                # Stream output to both console and log file.
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace",
                                        bufsize=1, env=child_env)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    logf.write(line)
                proc.wait()
                rc = proc.returncode
        except KeyboardInterrupt:
            print("\n  Interrupted by user. Stopping.")
            summary.append((tag, "interrupted", time.time() - t0))
            break
        except Exception as e:
            print(f"  ERROR launching run: {e}")
            rc = -1

        dt = time.time() - t0

        if rc == 0:
            # Rename results.json -> results.<model>.json so the next model on
            # the same pair does not overwrite it.
            produced = os.path.join(rdir, "results.json")
            if os.path.isfile(produced):
                shutil.move(produced, final_result)
                print(f"  OK ({dt/60:.1f} min) — results -> {final_result}")
                summary.append((tag, "ok", dt))
            else:
                print(f"  DONE ({dt/60:.1f} min) but no results.json found in {rdir}")
                summary.append((tag, "ok-no-result", dt))
        else:
            print(f"  FAILED (rc={rc}, {dt/60:.1f} min) — see {log_path}")
            summary.append((tag, f"failed(rc={rc})", dt))
            if not args.continue_on_error:
                print("\n  Stopping (use --continue-on-error to keep going).")
                break

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  SUMMARY  (total {(time.time()-t_start)/60:.1f} min)\n{'='*70}")
    for tag, status, dt in summary:
        mins = f"{dt/60:5.1f}m" if dt else "   -- "
        print(f"  {status:16s} {mins}  {tag}")
    n_ok = sum(1 for _, s, _ in summary if s == "ok")
    print(f"\n  {n_ok}/{len(summary)} runs completed successfully.\n")


if __name__ == "__main__":
    main()
