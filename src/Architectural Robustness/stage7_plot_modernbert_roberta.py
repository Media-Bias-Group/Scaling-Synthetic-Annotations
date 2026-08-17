"""plot_arch.py — four-way scaling comparison (lexical bias, mixed 50:50).
DeBERTa-base = muted reference; DeBERTa-large, RoBERTa, ModernBERT = models under test."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------- data
BASE = Path("/results")
df = pd.read_csv(BASE / "arch_ablation_summary.csv")          # roberta + modernbert

# DeBERTa-base (reference) from the earlier main summary
MAIN = Path("/results/stage5_summary.csv")
deb = pd.read_csv(MAIN)
deb = deb[(deb["task"] == "lexical_bias") & (deb["source"] == "mixed")].copy()
deb["model"] = "deberta-v3-base"

# DeBERTa-large (new) from its own summary
DEBLARGE = Path("/results/deberta_large_summary.csv")
debl = pd.read_csv(DEBLARGE)                                  # model col already 'deberta-v3-large'

cols = ["model", "size", "gold_f1_mean", "gold_f1_std",
        "bench_f1_mean", "bench_f1_std", "gold_mcc_mean", "bench_mcc_mean"]
def keep(d): return d[[c for c in cols if c in d.columns]]
df = pd.concat([keep(df), keep(deb), keep(debl)], ignore_index=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})

STYLE = {
    # DeBERTa family = blue pair (base muted reference, large bold)
    "deberta-v3-base":  dict(color="#9ecae1", ls="--", lw=1.4, ms=4, marker="o",
                             alpha=0.75, zorder=1, band=0.08,
                             label="DeBERTa-v3 base (reference)"),
    "deberta-v3-large": dict(color="#08519c", ls="-",  lw=2.0, ms=6, marker="D",
                             alpha=1.0, zorder=4, band=0.15,
                             label="DeBERTa-v3 large"),
    "roberta-base":     dict(color="#d62728", ls="-",  lw=2.0, ms=6, marker="o",
                             alpha=1.0, zorder=3, band=0.15, label="RoBERTa"),
    "ModernBERT-base":  dict(color="#2ca02c", ls="-",  lw=2.0, ms=6, marker="s",
                             alpha=1.0, zorder=3, band=0.15, label="ModernBERT"),
}
# draw reference first (underneath), models on top
ORDER = ["deberta-v3-base", "roberta-base", "ModernBERT-base", "deberta-v3-large"]
EVAL = [("gold_f1", "In-domain (gold)"), ("bench_f1", "Out-of-distribution")]

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)
for ax, (metric, title) in zip(axes, EVAL):
    for m in ORDER:
        s = df[df.model == m].sort_values("size")
        if s.empty or f"{metric}_mean" not in s.columns:
            continue
        st = STYLE[m]
        x = s["size"].values
        mean = s[f"{metric}_mean"].values
        sd = s[f"{metric}_std"].values if f"{metric}_std" in s.columns else None
        ax.plot(x, mean, ls=st["ls"], marker=st["marker"], color=st["color"],
                lw=st["lw"], ms=st["ms"], alpha=st["alpha"], zorder=st["zorder"],
                label=st["label"])
        if sd is not None:
            ax.fill_between(x, mean - sd, mean + sd, color=st["color"],
                            alpha=st["band"], lw=0, zorder=st["zorder"] - 1)
    ax.set_xscale("log")
    ax.set_xticks([1000, 4000, 16000, 64000, 128000])
    ax.set_xticklabels(["1k", "4k", "16k", "64k", "128k"])
    ax.set_xlabel("Training set size (log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)

# legend: models first, reference last
handles, labels = axes[0].get_legend_handles_labels()
leg_order = ["deberta-v3-large", "roberta-base", "ModernBERT-base", "deberta-v3-base"]
idx = [labels.index(STYLE[m]["label"]) for m in leg_order if STYLE[m]["label"] in labels]
fig.legend([handles[i] for i in idx], [labels[i] for i in idx],
           loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.05), fontsize=10.5)
fig.suptitle("Architecture & scale robustness: scaling on lexical bias (mixed 50:50 source)",
             fontsize=13, y=1.14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("pdf", "png"):
    fig.savefig(BASE / f"fig_arch_scaling2.{ext}", bbox_inches="tight", dpi=300)
print("wrote fig_arch_scaling2.pdf / .png")
