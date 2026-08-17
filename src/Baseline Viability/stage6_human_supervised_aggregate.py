"""stage6_human_supervised_aggregate.py — aggregate the B4 human_supervised folds.

Reads the per-fold JSONs written by stage6_human_supervised_train.py
(stage6_human_supervised_results/{task}_human_supervised_fold{fold}.json) and emits
stage6_human_supervised_summary.csv with, per task:
  * gold macro-F1 + MCC : 6-fold mean/std/ci95
  * bench macro-F1 + MCC: 6-fold mean/std/ci95
  * n_collapsed         : #folds whose gold eval collapsed to one class
  * n_folds             : folds found (expect 6)
  * bench_n             : benchmark rows scored (full benchmark; parity flag
                          for Baseline_comparison.py's bench_n column)

The output row carries condition="B4_human_supervised" and column names aligned with
the student summary (gold_f1_mean/std, gold_mcc_mean/std, bench_*), so it can
be concatenated straight into Baseline_comparison.py as the B4 column.

Usage:
  python stage6_human_supervised_aggregate.py
  python stage6_human_supervised_aggregate.py --results ~/twitter_pipeline/data/stage6_human_supervised_results
"""
import argparse, json, glob
import numpy as np
import polars as pl

from stage5_config import DATA, N_FOLDS, TASKS


def ci95(values):
    """95% CI half-width across folds (mirrors stage5d_aggregate.ci95)."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) > 1:
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return 0.0


def load_all(results_dir):
    rows = []
    for f in glob.glob(str(results_dir / "*_human_supervised_fold*.json")):
        d = json.load(open(f))
        rows.append({
            "task":             d["task"],
            "source":           d.get("source", "gold_human"),
            "condition":        d.get("condition", "B4_human_supervised"),
            "fold":             d["fold"],
            "n_train":          d.get("n_train"),
            "n_gold_eval":      d.get("n_gold_eval"),
            "n_benchmark":      d.get("n_benchmark"),
            "gold_macro_f1":    d.get("gold_macro_f1"),
            "gold_mcc":         d.get("gold_mcc"),
            "gold_accuracy":    d.get("gold_accuracy"),
            "bench_macro_f1":   d.get("bench_macro_f1"),
            "bench_mcc":        d.get("bench_mcc"),
            "heldout_macro_f1": d.get("heldout_macro_f1"),
            "gold_collapsed":   d.get("gold_collapsed", False),
            "bench_collapsed":  d.get("bench_collapsed", False),
            "final_train_loss": d.get("final_train_loss"),
            "train_minutes":    d.get("train_minutes"),
        })
    if not rows:
        raise SystemExit(f"No human_supervised result JSONs found in {results_dir}")
    return pl.DataFrame(rows)


def summarize(df):
    """Per-task 6-fold aggregation. ci95 is computed via a Python map because
    it is not a native polars agg; mean/std use polars directly (as in 5d)."""
    g = (df.group_by(["task", "condition", "source"])
           .agg([
               # ---- gold (in-domain) ----
               pl.col("gold_macro_f1").mean().alias("gold_f1_mean"),
               pl.col("gold_macro_f1").std().alias("gold_f1_std"),
               pl.col("gold_mcc").mean().alias("gold_mcc_mean"),
               pl.col("gold_mcc").std().alias("gold_mcc_std"),
               pl.col("gold_accuracy").mean().alias("gold_acc_mean"),
               # ---- bench (OOD, full benchmark) ----
               pl.col("bench_macro_f1").mean().alias("bench_f1_mean"),
               pl.col("bench_macro_f1").std().alias("bench_f1_std"),
               pl.col("bench_mcc").mean().alias("bench_mcc_mean"),
               pl.col("bench_mcc").std().alias("bench_mcc_std"),
               # ---- fidelity / bookkeeping ----
               pl.col("heldout_macro_f1").mean().alias("heldout_f1_mean"),
               pl.col("gold_collapsed").sum().alias("n_collapsed"),
               pl.col("bench_collapsed").sum().alias("n_bench_collapsed"),
               pl.len().alias("n_folds"),
               # full benchmark => bench_n identical across folds; take max
               pl.col("n_benchmark").max().alias("bench_n"),
               pl.col("n_train").max().alias("n_train"),
               pl.col("n_gold_eval").max().alias("n_gold_eval"),
               pl.col("train_minutes").sum().alias("total_train_min"),
               # collect raw fold values for ci95 (computed below)
               pl.col("gold_macro_f1").alias("_gold_f1_vals"),
               pl.col("gold_mcc").alias("_gold_mcc_vals"),
               pl.col("bench_macro_f1").alias("_bench_f1_vals"),
               pl.col("bench_mcc").alias("_bench_mcc_vals"),
           ])
           .sort(["task"]))

    # ci95 half-widths (list columns -> scalar), then drop the raw list cols
    g = g.with_columns([
        pl.col("_gold_f1_vals").map_elements(ci95, return_dtype=pl.Float64)
            .alias("gold_f1_ci95"),
        pl.col("_gold_mcc_vals").map_elements(ci95, return_dtype=pl.Float64)
            .alias("gold_mcc_ci95"),
        pl.col("_bench_f1_vals").map_elements(ci95, return_dtype=pl.Float64)
            .alias("bench_f1_ci95"),
        pl.col("_bench_mcc_vals").map_elements(ci95, return_dtype=pl.Float64)
            .alias("bench_mcc_ci95"),
    ]).drop(["_gold_f1_vals", "_gold_mcc_vals",
             "_bench_f1_vals", "_bench_mcc_vals"])

    # column order shaped for Baseline_comparison.py's B4 column
    ordered = [
        "task", "condition", "source", "n_folds",
        "gold_f1_mean", "gold_f1_std", "gold_f1_ci95",
        "gold_mcc_mean", "gold_mcc_std", "gold_mcc_ci95",
        "gold_acc_mean",
        "bench_f1_mean", "bench_f1_std", "bench_f1_ci95",
        "bench_mcc_mean", "bench_mcc_std", "bench_mcc_ci95",
        "heldout_f1_mean",
        "n_collapsed", "n_bench_collapsed", "bench_n",
        "n_train", "n_gold_eval", "total_train_min",
    ]
    return g.select([c for c in ordered if c in g.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DATA / "stage6_human_supervised_results"),
                    help="dir of {task}_human_supervised_fold{fold}.json files")
    ap.add_argument("--out", default=str(DATA / "stage6_human_supervised_summary.csv"))
    args = ap.parse_args()

    from pathlib import Path
    results_dir = Path(args.results)

    df = load_all(results_dir)
    summary = summarize(df)

    # ---- console report ----
    print(summary)
    n_expected = len(TASKS) * N_FOLDS
    print(f"\nhuman_supervised folds found: {df.height} (expect {n_expected} "
          f"= {len(TASKS)} tasks x {N_FOLDS} folds)")
    total_collapsed = int(df["gold_collapsed"].sum())
    if total_collapsed:
        print(f"[!! WARN] {total_collapsed} human_supervised fold(s) collapsed on gold — "
              f"inspect before trusting the B4 upper bound.")
    # per-task completeness + any missing folds
    for t in TASKS:
        got = df.filter(pl.col("task") == t).height
        if got != N_FOLDS:
            print(f"[!! WARN] task '{t}': found {got}/{N_FOLDS} folds.")

    summary.write_csv(args.out)
    print(f"\nWrote {args.out}")
    print("Drop this into Baseline_comparison.py as the B4_human_supervised column "
          "(gold_f1_mean/std + gold_mcc_mean/std; bench_* for RQ2; "
          "bench_n flags full-vs-5k parity).")


if __name__ == "__main__":
    main()
