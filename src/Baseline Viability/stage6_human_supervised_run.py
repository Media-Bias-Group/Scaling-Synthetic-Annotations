"""Stage 6 — Orchestrate the B4 human-supervised human_supervised sweep.

Analogue of stage5c_run_sweep.py, but for the human_supervised: no size loop, no
source loop (one human_supervised per task on the full gold_train partition). Loops
only over (task, fold) and calls stage6_human_supervised_train.py as a subprocess.

Resumable: skips any run whose per-fold JSON already exists (same
skip-if-done pattern as stage5c_run_sweep). Robust to preemption on
spot/community-cloud pods — a killed pod loses at most one fold.

Grid: 4 tasks x 6 folds = 24 runs.

Usage:
  python stage6_human_supervised_run.py                 # run everything not yet done
  python stage6_human_supervised_run.py --task sexism   # restrict to one task
  python stage6_human_supervised_run.py --dry-run       # list what would run
"""
import argparse, subprocess, sys, time
from pathlib import Path

from stage5_config import TASKS, N_FOLDS, DATA

human_supervised_DIR = DATA / "stage6_human_supervised_results"


def done(task, fold):
    p = human_supervised_DIR / f"{task}_human_supervised_fold{fold}.json"
    return p.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, choices=list(TASKS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = [args.task] if args.task else list(TASKS)

    conditions = [(t, f) for t in tasks for f in range(N_FOLDS)]
    todo = [c for c in conditions if not done(*c)]
    print(f"Total conditions: {len(conditions)} | "
          f"already done: {len(conditions) - len(todo)} | "
          f"to run: {len(todo)}")

    if args.dry_run:
        for c in todo:
            print("  would run:", c)
        return

    # resolve trainer next to this orchestrator, so cwd doesn't matter
    trainer = str(Path(__file__).resolve().parent / "stage6_human_supervised_train.py")

    t0 = time.time()
    for i, (t, f) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] task={t} fold={f}")
        cmd = [sys.executable, trainer, "--task", t, "--fold", str(f)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  !! run failed (rc={rc}); leaving un-done so it retries "
                  f"next launch.")
        el = (time.time() - t0) / 60
        print(f"  elapsed {el:.1f} min total")

    print("\nhuman_supervised sweep complete.")


if __name__ == "__main__":
    main()
