
"""
Results

Uses stage5_summary.csv (per-fold aggregates: *_mean, *_std over n_folds=6).
Rows with source starting 'abl_' are RQ-B ablations and are excluded.
"""

import numpy as np
import pandas as pd

CSV   = "/results/stage5_summary.csv"
SIZES = [1000, 4000, 16000, 64000, 128000]
EVALS = {"gold": "gold_f1_mean", "bench": "bench_f1_mean", "heldout": "heldout_f1_mean"}
STDS  = {"gold": "gold_f1_std",  "bench": "bench_f1_std"}
ORDER = ["twitter", "news", "mixed"]

# --- Authoritative trend tests from the slides (raw-fold Friedman + Page's) ---
# (task, source, eval): (friedman_p, pages_L, pages_p, verdict)
TREND = {
    ("hate","twitter","gold"):        (0.0001, 329.0, "<0.0001", "Monotonic ↑"),
    ("hate","twitter","bench"):       (0.0001, 272.0, "0.4522",  "No trend ⚠"),
    ("lexical_bias","twitter","gold"):(0.0009, 312.0, "0.0001",  "Monotonic ↑"),
    ("lexical_bias","twitter","bench"):(0.0041,315.0, "<0.0001", "Monotonic ↑"),
    ("sentiment","twitter","gold"):   (0.0007, 322.0, "<0.0001", "Monotonic ↑"),
    ("sentiment","twitter","bench"):  (0.0002, 315.0, "<0.0001", "Monotonic ↑"),
    ("sexism","twitter","gold"):      (0.0001, 329.0, "<0.0001", "Monotonic ↑"),
    ("sexism","twitter","bench"):     (0.0036, 314.0, "0.0001",  "Monotonic ↑"),
    ("lexical_bias","news","gold"):   (0.0009, 308.0, "0.0006",  "Monotonic ↑"),
    ("lexical_bias","news","bench"):  (0.0005, 324.0, "<0.0001", "Monotonic ↑"),
    ("sentiment","news","gold"):      (0.0003, 320.0, "<0.0001", "Monotonic ↑"),
    ("sentiment","news","bench"):     (0.0146, 275.0, "0.3592",  "No trend ⚠"),
    ("lexical_bias","mixed","gold"):  (0.0002, 327.0, "<0.0001", "Monotonic ↑"),
    ("lexical_bias","mixed","bench"): (0.0003, 324.0, "<0.0001", "Monotonic ↑"),
    ("sentiment","mixed","gold"):     (0.0003, 326.0, "<0.0001", "Monotonic ↑"),
    ("sentiment","mixed","bench"):    (0.0116, 300.0, "0.0069",  "Monotonic ↑"),
}

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def load():
    df = pd.read_csv(CSV)
    df = df[~df["source"].str.startswith("abl_")].copy()
    df["size"] = df["size"].astype(int)
    return df.sort_values(["task", "source", "size"]).reset_index(drop=True)


# ---------------------------------------------------------------- 1. curves
def scaling_curves(df):
    print("\n" + "=" * 100)
    print("1. SCALING CURVES  -- macro-F1 mean (std) at each N")
    print("=" * 100)
    for (task, source), g in df.groupby(["task", "source"]):
        g = g.set_index("size").reindex(SIZES)
        disp = pd.DataFrame(index=SIZES)
        for ev, col in EVALS.items():
            if ev in STDS:
                disp[ev] = [f"{m:.4f} ({s:.4f})" if pd.notna(m) else "--"
                            for m, s in zip(g[col], g[STDS[ev]])]
            else:
                disp[ev] = [f"{m:.4f}" if pd.notna(m) else "--" for m in g[col]]
        disp.index.name = "N"
        print(f"\n--- {task} | {source} ---")
        print(disp.to_string())


# ---------------------------------------------------------------- 2. gains/peaks
def gains_and_peaks(df):
    rows = []
    for (task, source), g in df.groupby(["task", "source"]):
        g = g.set_index("size")
        for ev, col in EVALS.items():
            s = g[col].reindex(SIZES).dropna()
            if s.empty:
                continue
            lo, hi, peak = s.index.min(), s.index.max(), s.idxmax()
            rows.append(dict(
                task=task, source=source, eval=ev,
                f1_1k=round(s.loc[lo], 4),
                f1_128k=round(s.loc[hi], 4),
                abs_gain=round(s.loc[hi] - s.loc[lo], 4),
                peak_N=int(peak), peak_f1=round(s.loc[peak], 4),
                drop_after_peak=round(s.loc[peak] - s.loc[hi], 4),
            ))
    res = pd.DataFrame(rows).sort_values(["eval", "task", "source"])
    print("\n" + "=" * 100)
    print("2. GAINS (1k->128k) & DIMINISHING RETURNS")
    print("   drop_after_peak>0  =>  performance decreases after the peak size")
    print("=" * 100)
    print(res.to_string(index=False))
    return res


# ---------------------------------------------------------------- 3. cross-source
def cross_source(df):
    print("\n" + "=" * 100)
    print("3. CROSS-SOURCE COMPARISON  (lexical_bias & sentiment have all 3 sources)")
    print("   'bench' = OOD benchmark: the key panel for the diversity claim")
    print("=" * 100)
    piv_store = {}
    for ev, col in EVALS.items():
        piv = df.pivot_table(index=["task", "size"], columns="source", values=col)
        piv = piv[[c for c in ORDER if c in piv.columns]]
        piv_store[ev] = piv
        print(f"\n--- eval = {ev} ---")
        print(piv.round(4).to_string())

    # Explicit "best source" per (task,size) on OOD, for the multi-source tasks
    print("\n--- OOD winner per (task, size) among {twitter,news,mixed} ---")
    b = piv_store["bench"].dropna(how="all")
    multi = b.dropna(axis=0, thresh=2)   # rows with >=2 sources present
    if not multi.empty:
        win = multi.idxmax(axis=1)
        best = multi.max(axis=1).round(4)
        margin = (multi.max(axis=1) - multi.drop(columns=[]).apply(
            lambda r: r.sort_values(ascending=False).iloc[1], axis=1)).round(4)
        summ = pd.DataFrame({"winner": win, "best_f1": best, "margin_over_2nd": margin})
        print(summ.to_string())


# ---------------------------------------------------------------- 4. trend table
def trend_table():
    print("\n" + "=" * 100)
    print("4. AUTHORITATIVE FRIEDMAN + PAGE'S TREND TESTS (from slides, raw folds)")
    print("=" * 100)
    rows = [dict(task=t, source=s, eval=e, friedman_p=fp,
                 pages_L=L, pages_p=pp, verdict=v)
            for (t, s, e), (fp, L, pp, v) in TREND.items()]
    res = pd.DataFrame(rows).sort_values(["source", "task", "eval"])
    print(res.to_string(index=False))


def main():
    df = load()
    print(f"Loaded {len(df)} non-ablation rows | tasks={sorted(df.task.unique())} "
          f"| sources={sorted(df.source.unique())}")
    scaling_curves(df)
    gains_and_peaks(df)
    cross_source(df)
    trend_table()


if __name__ == "__main__":
    main()
