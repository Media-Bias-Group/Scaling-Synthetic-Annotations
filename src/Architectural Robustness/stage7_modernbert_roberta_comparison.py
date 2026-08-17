
import json, glob, re
import pandas as pd
from pathlib import Path

BASE = Path("/results")
SELECTED_LR = {"roberta-base": "3e-5", "ModernBERT-base": "2e-5"}

# filename: lexical_bias_mixed_{size}_fold{fold}_{model}_lr{lr}.json
pat = re.compile(r"lexical_bias_mixed_(\d+)_fold(\d+)_(.+?)_lr([\d.e+-]+)\.json$")

rows = []
for f in glob.glob(str(BASE / "**" / "lexical_bias_mixed_*.json"), recursive=True):
    name = Path(f).name
    m = pat.match(name)
    if not m:
        print("skip (no match):", name); continue
    size, fold, model, lr = int(m[1]), int(m[2]), m[3], m[4]
    # keep only the selected LR for each model  -> drops the 4 LR-test leftovers
    if SELECTED_LR.get(model) != lr:
        continue
    d = json.load(open(f))
    rows.append(dict(model=model, size=size, fold=fold, lr=lr,
                     gold_f1=d.get("gold_macro_f1"), gold_mcc=d.get("gold_mcc"),
                     bench_f1=d.get("bench_macro_f1"), bench_mcc=d.get("bench_mcc")))

df = pd.DataFrame(rows)
print(f"kept {len(df)} rows (expect 60)")

# sanity: every (model,size) must have exactly 6 folds
counts = df.groupby(["model", "size"])["fold"].nunique().reset_index(name="n_folds")
print("Fold counts (should all be 6):")
print(counts.to_string(index=False))
assert (counts["n_folds"] == 6).all(), "Some cell != 6 folds — check filter!"

agg = (df.groupby(["model", "size"])
         .agg(gold_f1_mean=("gold_f1","mean"), gold_f1_std=("gold_f1","std"),
              gold_mcc_mean=("gold_mcc","mean"), gold_mcc_std=("gold_mcc","std"),
              bench_f1_mean=("bench_f1","mean"), bench_f1_std=("bench_f1","std"),
              bench_mcc_mean=("bench_mcc","mean"), bench_mcc_std=("bench_mcc","std"),
              n_folds=("fold","nunique"))
         .reset_index())
agg.to_csv(BASE / "arch_ablation_summary.csv", index=False)
print("\nwrote arch_ablation_summary.csv")
print(agg.to_string(index=False))
