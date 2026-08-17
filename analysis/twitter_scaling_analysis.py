import json, os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, page_trend_test
from statsmodels.stats.multitest import multipletests   # <-- add this

RESULTS_DIR = "/results/stage5_results"
SIZES   = [1000, 4000, 16000, 64000, 128000]
SOURCES = ["twitter", "news", "mixed"]
TASKS   = ["hate", "lexical_bias", "sentiment", "sexism"]
N_FOLDS = 6

# --- 1. Load fold-level JSONs (unchanged) ---
rows = []
for task in TASKS:
    for source in SOURCES:
        for size in SIZES:
            for fold in range(N_FOLDS):
                fp = os.path.join(RESULTS_DIR, f"{task}_{source}_{size}_fold{fold}.json")
                if not os.path.exists(fp):
                    continue
                d = json.load(open(fp))
                rows.append({
                    "task": task, "source": source, "size": size,
                    "fold": fold, "seed": d.get("seed"),
                    "gold_f1":  d["gold_macro_f1"],
                    "bench_f1": d["bench_macro_f1"],
                    "gold_collapsed":  d.get("gold_collapsed", False),
                    "bench_collapsed": d.get("bench_collapsed", False),
                })
df = pd.DataFrame(rows)

# --- 2. Complete grids (unchanged) ---
def complete_grids(df):
    grids = []
    for (task, source), g in df.groupby(["task", "source"]):
        piv = g.pivot(index="fold", columns="size", values="gold_f1")
        have_sizes = set(piv.columns)
        if set(SIZES).issubset(have_sizes) and piv[SIZES].notna().all().all():
            grids.append((task, source))
        else:
            missing = set(SIZES) - have_sizes
            print(f"SKIP {task}/{source}: incomplete grid, missing sizes {sorted(missing)}")
    return grids

grids = complete_grids(df)

# --- 3. Sanity check (unchanged) ---
print("\n--- sanity ---")
for task, source in grids:
    g = df[(df.task == task) & (df.source == source)]
    piv = g.pivot(index="fold", columns="size", values="gold_f1")[SIZES]
    seed_ok = (g.groupby("fold")["seed"].nunique() == 1).all()
    n_coll  = int(g[["gold_collapsed", "bench_collapsed"]].sum().sum())
    print(f"{task:13s} {source:8s} cells={piv.shape} "
          f"missing={int(piv.isna().sum().sum())} seed_paired={seed_ok} collapsed={n_coll}")

# --- 4. Analyze: now RETURNS a record instead of only printing ---
def analyze(task, source, metric):
    g = df[(df.task == task) & (df.source == source)]
    sub = g.pivot(index="fold", columns="size", values=metric)[SIZES]
    assert sub.notna().all().all()

    means = {s: sub[s].mean() for s in SIZES}

    stat, p_f = friedmanchisquare(*[sub[s].values for s in SIZES])
    res = page_trend_test(sub.values, ranked=False, predicted_ranks=None)

    delta_f1 = means[SIZES[-1]] - means[SIZES[0]]        # 128k - 1k
    n_star   = max(means, key=means.get)                 # size of peak mean

    print(f"\n{'='*60}\n{task} | {source} | {metric} (n_folds={sub.shape[0]})\n{'='*60}")
    for s in SIZES:
        print(f"  size {s:>6}: mean={means[s]:.4f}  std={sub[s].std(ddof=1):.4f}")
    print(f"  Friedman chi2={stat:.3f}, p={p_f:.4f}")
    print(f"  Page's L={res.statistic:.1f}, p={res.pvalue:.4f}")
    print(f"  delta_f1(128k-1k)={delta_f1:+.3f}  N*={n_star}")

    return {
        "source": source, "task": task, "eval": metric,
        "friedman_p": p_f,
        "page_L": res.statistic,
        "page_p_raw": res.pvalue,
        "delta_f1": delta_f1,
        "N_star": n_star,
    }

records = []
for task, source in grids:
    for metric in ["gold_f1", "bench_f1"]:
        records.append(analyze(task, source, metric))

results = pd.DataFrame(records)

# --- 5. Holm correction across ALL Page's tests jointly ---
reject, p_holm, _, _ = multipletests(
    results["page_p_raw"].values, alpha=0.05, method="holm"
)
results["page_p_holm"] = p_holm
results["reject_holm"] = reject   # True where p_holm < 0.05

# --- 6. Verdict that distinguishes "rising" from "rising but peak < max" ---
def verdict(row):
    if not row["reject_holm"]:
        return "No trend"
    if row["N_star"] != SIZES[-1]:          # significant trend but peak before 128k
        return "Rising (peak < max)"
    return "Rising"

results["verdict"] = results.apply(verdict, axis=1)

# --- 7. Tidy output, ordered as in the manuscript table ---
eval_name = {"gold_f1": "gold", "bench_f1": "bench"}
results["eval"] = results["eval"].map(eval_name)
source_order = {"twitter": 0, "news": 1, "mixed": 2}
results["src_ord"] = results["source"].map(source_order)
results = results.sort_values(["src_ord", "task", "eval"]).drop(columns="src_ord")

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n\n===== FINAL TABLE (paste-ready values) =====")
print(results[[
    "source", "task", "eval", "friedman_p", "page_L",
    "page_p_raw", "page_p_holm", "delta_f1", "N_star", "verdict"
]].to_string(index=False))

# Optional: dump straight to CSV for the paper
results.to_csv(os.path.join(RESULTS_DIR, "scaling_trend_summary.csv"), index=False)
