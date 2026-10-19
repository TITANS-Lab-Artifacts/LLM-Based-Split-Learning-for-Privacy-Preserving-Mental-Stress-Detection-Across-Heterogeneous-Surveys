"""
run_inference_all.py
====================
Batch-runs Zero_Shot.py and Few_Shot.py over every train/test dataset pair,
for both models, one job at a time.

Pairs (same-suffix train<->test):
  train_hetero              <-> test_hetero
  train_mod_hetero          <-> test_mod_hetero
  train_mh_dass_dataset     <-> test_mh_dass_dataset
  train_mh_general_dataset  <-> test_mh_general_dataset
  train_mh_student_dataset  <-> test_mh_student_dataset
  train_mh_tech_dataset     <-> test_mh_tech_dataset

Models:
  meta-llama/Llama-3.2-3B-Instruct
  aaditya/Llama3-OpenBioLLM-8B

Default sweep = 2 models x 6 pairs x {zero_shot, few_shot}.
Few-shot uses --n_few 10. Generation batch size = 64.

The two inference scripts write their own results to <test_dir>/results/.
This runner just launches them with the right args and logs each run.

USAGE:
  python run_inference_all.py                  # everything (both modes, both models)
  python run_inference_all.py --modes few_shot # only few-shot
  python run_inference_all.py --models llama   # only the 3B model
  python run_inference_all.py --dry-run        # print commands only
  python run_inference_all.py --n_test 500     # cap test rows (quick pass)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# ── EDIT THESE IF NEEDED ──────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "processed", "text")   # folder with the .jsonl files
N_FEW     = 10           # few-shot examples per prompt
BATCH_SIZE = 32          # generation batch size
N_TEST    = None         # cap on test rows (None = all); CLI --n_test overrides

ZERO_SHOT_SCRIPT = "Zero_Shot.py"
FEW_SHOT_SCRIPT  = "Few_Shot.py"

# train file -> test file (same-suffix pairing)
PAIRS = [
    ("train_hetero.jsonl",             "test_hetero.jsonl"),
    ("train_mod_hetero.jsonl",         "test_mod_hetero.jsonl"),
    ("train_mh_dass_dataset.jsonl",    "test_mh_dass_dataset.jsonl"),
    ("train_mh_general_dataset.jsonl", "test_mh_general_dataset.jsonl"),
    ("train_mh_student_dataset.jsonl", "test_mh_student_dataset.jsonl"),
    ("train_mh_tech_dataset.jsonl",    "test_mh_tech_dataset.jsonl"),
]

# model key -> HF model id
MODELS = {
    "llama":   "meta-llama/Llama-3.2-3B-Instruct",
    "openbio": "aaditya/Llama3-OpenBioLLM-8B",
}
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR = "inference_logs"


def main():
    ap = argparse.ArgumentParser(description="Batch zero/few-shot inference over all pairs.")
    ap.add_argument("--modes", nargs="+", choices=["zero_shot", "few_shot"],
                    default=["zero_shot", "few_shot"],
                    help="Which evaluation mode(s) to run (default: both).")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                    help="Which model(s) to run (default: both).")
    ap.add_argument("--n_few", type=int, default=N_FEW,
                    help=f"Few-shot examples per prompt (default {N_FEW}).")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                    help=f"Generation batch size (default {BATCH_SIZE}).")
    ap.add_argument("--n_test", type=int, default=N_TEST,
                    help="Cap on test rows per run (default: all).")
    ap.add_argument("--gpu", type=str, default=None,
                    help="Set CUDA_VISIBLE_DEVICES for every run (e.g. '0' or '0,1'). "
                         "If unset, each script keeps its own default device.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running them.")
    ap.add_argument("--no-skip-existing", "--force", dest="no_skip_existing",
                    action="store_true",
                    help="Re-run configs even if their result JSON already exists. "
                         "By default, a config whose result file is present is skipped.")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going if a run fails (default: stop).")
    args = ap.parse_args()
    # Skipping existing results is the default; --no-skip-existing/--force disables it.
    args.skip_existing = not args.no_skip_existing

    os.makedirs(LOG_DIR, exist_ok=True)
    results_dir = os.path.join(DATA_DIR, "results")

    # Build the job list. Order: per pair, per model, per mode.
    jobs = []
    for tr, te in PAIRS:
        for mkey in args.models:
            for mode in args.modes:
                jobs.append((mode, mkey, tr, te))
    total = len(jobs)

    print(f"\n{'='*72}")
    print(f"  INFERENCE SWEEP — {len(args.models)} model(s) x {len(PAIRS)} pair(s) "
          f"x {len(args.modes)} mode(s) = {total} runs")
    print(f"  modes      : {', '.join(args.modes)}")
    print(f"  models     : {', '.join(args.models)}")
    print(f"  n_few      : {args.n_few}    batch_size: {args.batch_size}"
          f"    n_test: {args.n_test if args.n_test else 'all'}")
    if args.gpu is not None:
        print(f"  GPU pin    : CUDA_VISIBLE_DEVICES={args.gpu}")
    print(f"  skip done  : {'yes (existing result JSON -> skip)' if args.skip_existing else 'no (--force)'}")
    print(f"{'='*72}\n")

    # Warn about missing files (don't abort — let the user see the full list).
    needed = set()
    for tr, te in PAIRS:
        needed.add(te)
        if "few_shot" in args.modes:
            needed.add(tr)
    missing = [fn for fn in sorted(needed)
               if not os.path.isfile(os.path.join(DATA_DIR, fn))]
    for scr in (ZERO_SHOT_SCRIPT, FEW_SHOT_SCRIPT):
        if not os.path.isfile(scr):
            missing.append(scr)
    if missing:
        print("  WARNING — files not found in the current directory:")
        for fn in missing:
            print(f"    - {fn}")
        print()

    def expected_result(mode, mkey, tr, te):
        """Mirror the output filenames the two scripts produce."""
        model_slug = MODELS[mkey].replace("/", "_").replace("-", "_")
        tr_stem = os.path.splitext(tr)[0]
        te_stem = os.path.splitext(te)[0]
        if mode == "zero_shot":
            fn = f"zero_shot_{model_slug}_{te_stem}.json"
        else:
            fn = f"few_shot_{args.n_few}_{model_slug}_{tr_stem}_vs_{te_stem}.json"
        return os.path.join(results_dir, fn)

    summary = []
    t_start = time.time()

    for idx, (mode, mkey, tr, te) in enumerate(jobs, 1):
        model_id = MODELS[mkey]
        tag = f"[{idx}/{total}] {mode:9s} {mkey:8s} {te}"
        print(f"\n{'-'*72}\n  {tag}\n{'-'*72}")

        res_path = expected_result(mode, mkey, tr, te)
        if args.skip_existing and os.path.isfile(res_path):
            print(f"  SKIP — {res_path} already exists.")
            summary.append((tag, "skipped", 0.0))
            continue

        # Build the command for this mode.
        if mode == "zero_shot":
            cmd = [sys.executable, ZERO_SHOT_SCRIPT,
                   "--test", os.path.join(DATA_DIR, te),
                   "--model", model_id,
                   "--batch_size", str(args.batch_size)]
        else:  # few_shot
            cmd = [sys.executable, FEW_SHOT_SCRIPT,
                   "--train", os.path.join(DATA_DIR, tr),
                   "--test",  os.path.join(DATA_DIR, te),
                   "--model", model_id,
                   "--n_few", str(args.n_few),
                   "--batch_size", str(args.batch_size)]
        if args.n_test:
            cmd += ["--n_test", str(args.n_test)]

        print("  CMD:", " ".join(cmd))

        if args.dry_run:
            summary.append((tag, "dry-run", 0.0))
            continue

        # Child environment: force UTF-8 (Windows cp1252 breaks on box chars),
        # and optionally pin the GPU for every run.
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        if args.gpu is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = args.gpu

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(
            LOG_DIR, f"{mode}__{mkey}__{os.path.splitext(te)[0]}__{stamp}.log")
        print(f"  LOG: {log_path}")

        t0 = time.time()
        rc = 0
        try:
            with open(log_path, "w", encoding="utf-8") as logf:
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
            ok = os.path.isfile(res_path)
            print(f"  {'OK' if ok else 'DONE (result file not found?)'} "
                  f"({dt/60:.1f} min)")
            summary.append((tag, "ok" if ok else "ok-no-result", dt))
        else:
            print(f"  FAILED (rc={rc}, {dt/60:.1f} min) — see {log_path}")
            summary.append((tag, f"failed(rc={rc})", dt))
            if not args.continue_on_error:
                print("\n  Stopping (use --continue-on-error to keep going).")
                break

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  SUMMARY  (total {(time.time()-t_start)/60:.1f} min)\n{'='*72}")
    for tag, status, dt in summary:
        mins = f"{dt/60:5.1f}m" if dt else "   -- "
        print(f"  {status:16s} {mins}  {tag}")
    n_ok = sum(1 for _, s, _ in summary if s == "ok")
    print(f"\n  {n_ok}/{len(summary)} runs completed successfully.")
    print(f"  Result JSONs in: {results_dir}\n")


if __name__ == "__main__":
    main()