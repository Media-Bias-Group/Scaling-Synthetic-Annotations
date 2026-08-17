"""Stage 6 — five-condition comparison: student (S) vs Mistral/Gemma/Sonnet.

Tables: student best-config (6-fold mean+/-std) + LLM teachers in BOTH regimes
        (fewshot / zeroshot), macro-F1 and MCC, gold + benchmark.
.
"""
from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

RES = Path.home() / "/results"
STUDENT_CSV = Path.home() / "results/stage5_summary.csv"

STUDENT_CSV = Path("/results/stage5_summary.csv")
LLM_CSV = RES / "stage6_baseline_summary.csv"
FIG_DIR = RES / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TASKS = ["hate", "sexism", "sentiment", "lexical_bias"]
TASK_LABEL = {"hate": "Hate Speech", "sexism": "Sexism",
              "sentiment": "Sentiment", "lexical_bias": "Lexical Bias"}
STUDENT_SELECT_METRIC = "gold_f1_mean"   # best-config selection metric


OUT_DIR = Path("/results/stage6_baselines/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
})

TASKS = ["hate", "sexism", "sentiment", "lexical_bias"]
TASK_SHORT = {"hate": "Hate", "sexism": "Sexism",
              "sentiment": "Sentiment", "lexical_bias": "Lexical Bias"}

# Group zero-shot first, then student, then few-shot
rows = [("mistral", "zeroshot", "Mistral zs"),
        ("gemma", "zeroshot", "Gemma zs"),
        ("sonnet", "zeroshot", "Sonnet zs"),
        ("student", "supervised-distilled", "Student"),
        ("mistral", "fewshot", "Mistral fs"),
        ("gemma", "fewshot", "Gemma fs"),
        ("sonnet", "fewshot", "Sonnet fs")]

# seven entries, matched to the new ordering
bars = [("mistral", "zeroshot"),
        ("gemma", "zeroshot"),
        ("sonnet", "zeroshot"),
        ("student", "supervised-distilled"),
        ("mistral", "fewshot"),
        ("gemma", "fewshot"),
        ("sonnet", "fewshot")]

xt = ["Mis\nzs", "Gem\nzs", "Son\nzs", "Stud",
      "Mis\nfs", "Gem\nfs", "Son\nfs"]

def load_student():
    s = pl.read_csv(STUDENT_CSV)
    best = (s.sort(STUDENT_SELECT_METRIC, descending=True)
              .group_by("task", maintain_order=True).first())
    rows_data = []
    for t in TASKS:
        r = best.filter(pl.col("task") == t)
        if r.height == 0:
            print(f"  !! no student row for {t}")
            continue
        r = r.row(0, named=True)
        rows_data.append({
            "model": "student", "task": t,
            "config": f"{r['source']}/{r['size']}", "params": "184M",
            "gold_f1": r["gold_f1_mean"], "gold_f1_err": r["gold_f1_std"],
            "bench_f1": r["bench_f1_mean"], "bench_f1_err": r["bench_f1_std"],
            "gold_mcc": r.get("gold_mcc_mean"), "gold_mcc_err": r.get("gold_mcc_std"),
            "bench_mcc": r.get("bench_mcc_mean"), "bench_mcc_err": r.get("bench_mcc_std"),
            "bench_n": None, "regime": "supervised-distilled",
        })
    return rows_data


def load_llms():
    l = pl.read_csv(LLM_CSV)
    rows_data = []
    for regime in ["fewshot", "zeroshot"]:
        lr = l.filter(pl.col("regime") == regime)
        for model in ["mistral", "gemma", "sonnet"]:
            for t in TASKS:
                g = lr.filter((pl.col("model") == model) & (pl.col("task") == t)
                              & (pl.col("eval_set") == "gold"))
                b = lr.filter((pl.col("model") == model) & (pl.col("task") == t)
                              & (pl.col("eval_set") == "benchmark"))
                if g.height == 0 or b.height == 0:
                    print(f"  !! missing {model}/{t}/{regime}")
                    continue
                g, b = g.row(0, named=True), b.row(0, named=True)
                half = lambda hi, lo: (hi - lo) / 2
                rows_data.append({
                    "model": model, "task": t, "config": regime,
                    "params": g["params"], "regime": regime,
                    "gold_f1": g["honest_macro_f1"],
                    "gold_f1_err": half(g["honest_macro_f1_hi"], g["honest_macro_f1_lo"]),
                    "bench_f1": b["honest_macro_f1"],
                    "bench_f1_err": half(b["honest_macro_f1_hi"], b["honest_macro_f1_lo"]),
                    "gold_mcc": g["honest_mcc"],
                    "gold_mcc_err": half(g["honest_mcc_hi"], g["honest_mcc_lo"]),
                    "bench_mcc": b["honest_mcc"],
                    "bench_mcc_err": half(b["honest_mcc_hi"], b["honest_mcc_lo"]),
                    "bench_n": b["n"],
                })
    return rows_data


def print_tables(df):
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    for t in TASKS:
        print("\n" + "=" * 90)
        print(f"TASK: {TASK_LABEL[t]}  — macro-F1 (gold=in-domain, bench=OOD)")
        sub = df.filter(pl.col("task") == t).select(
            ["model", "regime", "config", "params",
             "gold_f1", "gold_f1_err", "bench_f1", "bench_f1_err", "bench_n"])
        print(sub)
        print(f"\nTASK: {TASK_LABEL[t]}  — MCC")
        subm = df.filter(pl.col("task") == t).select(
            ["model", "regime", "gold_mcc", "gold_mcc_err",
             "bench_mcc", "bench_mcc_err"])
        print(subm)


def make_figure(df, metric="f1"):
    gold_col, bench_col = f"gold_{metric}", f"bench_{metric}"
    gold_err, bench_err = f"gold_{metric}_err", f"bench_{metric}_err"

    # Uses the updated bars order
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(13, 3.6 * len(TASKS)),
                             sharey=True)

    CB = {"mistral": "#1f77b4", "gemma": "#ff7f0e", "sonnet": "#2ca02c", "student": "#d62728"}
    DISP = {"mistral": "Mistral", "gemma": "Gemma", "sonnet": "Sonnet", "student": "Student"}

    for ti, t in enumerate(TASKS):
        for ci, (evalset, vcol, ecol) in enumerate(
                [("In-domain (gold)", gold_col, gold_err),
                 ("OOD benchmark", bench_col, bench_err)]):
            ax = axes[ti, ci]
            vals, errs, colors, hatches, local_xt = [], [], [], [], []
            for (model, regime), x_label in zip(bars, xt):
                r = df.filter((pl.col("model") == model) & (pl.col("task") == t)
                              & (pl.col("regime") == regime))
                if r.height == 0:
                    vals.append(np.nan); errs.append(0)
                else:
                    r = r.row(0, named=True)
                    vals.append(r[vcol]); errs.append(r[ecol] or 0)
                colors.append(CB[model])
                hatches.append("//" if regime == "zeroshot" else "")
                local_xt.append(x_label)
            x = np.arange(len(bars))
            b = ax.bar(x, vals, yerr=errs, capsize=3, color=colors,
                       edgecolor="black", linewidth=0.7)
            for bar, h in zip(b, hatches):
                bar.set_hatch(h)
            ax.set_xticks(x); ax.set_xticklabels(local_xt, fontsize=12)
            ax.set_ylim(0, 1.0)
            if ci == 0:
                ax.set_ylabel(f"{TASK_LABEL[t]}\n{'Macro-F1' if metric=='f1' else 'MCC'}")
            if ti == 0:
                ax.set_title(evalset)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CB[m], ec="black")
               for m in ["mistral", "gemma", "sonnet", "student"]]
    labels = [DISP[m] for m in ["mistral", "gemma", "sonnet", "student"]]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch="//"))
    labels.append("zero-shot")
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.015), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    stem = FIG_DIR / f"stage6_comparison_{metric}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.pdf / .png")

def make_heatmap_table(df, out="stage6_heatmap_table"):
    cols = [(t, ev) for t in TASKS for ev in ["gold_f1", "bench_f1"]]
    col_labels = []
    for t in TASKS:
        col_labels += [f"{TASK_SHORT[t]}\nID", f"{TASK_SHORT[t]}\nOOD"]

    M = np.full((len(rows), len(cols)), np.nan)
    for i, (model, regime, _) in enumerate(rows):
        for j, (t, metric) in enumerate(cols):
            r = df.filter((pl.col("model") == model) & (pl.col("task") == t)
                          & (pl.col("regime") == regime))
            if r.height:
                M[i, j] = r.row(0, named=True)[metric]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    im = ax.imshow(M, cmap="YlGn", vmin=0.45, vmax=0.95, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows], fontsize=10)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()

    # thin separators between tasks
    for k in range(2, len(cols), 2):
        ax.axvline(k - 0.5, color="white", lw=2.5)

    # Borders above and below the student row to highlight it
    # Student is now at index 3, so lines at 2.5 and 3.5 frame it perfectly.
    ax.axhline(2.5, color="black", lw=1.2)
    ax.axhline(3.5, color="black", lw=1.2)

    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                weight = "bold" if rows[i][0] == "student" else "normal"
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight=weight,
                        color="black" if M[i, j] < 0.80 else "black")

    fig.tight_layout()
    stem = OUT_DIR / out
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / {stem}.png")

def main():
    df = pl.DataFrame(load_student() + load_llms(), infer_schema_length=None)
    order = {"mistral": 0, "gemma": 1, "sonnet": 2, "student": 3}
    df = df.with_columns(pl.col("model").replace_strict(order).alias("_o")) \
           .sort(["task", "_o", "regime"]).drop("_o")
    print_tables(df)
    df.write_csv(RES / "stage6_comparison_full.csv")
    print(f"\nWrote {RES / 'stage6_comparison_full.csv'}")

    make_heatmap_table(df)
    return df

def print_teacher_stats(df):
    print("\n" + "="*80)
    print("ABLATION 2 STATS — teacher performance (paste this back)")
    print("="*80)

    # 1. Full teacher table: F1 per model x regime x task x eval
    print("\n--- TEACHER MACRO-F1 (fs=fewshot, zs=zeroshot) ---")
    for t in TASKS:
        print(f"\n{TASK_LABEL[t]}:")
        for model in ["mistral", "gemma", "sonnet"]:
            for regime in ["fewshot", "zeroshot"]:
                r = df.filter((pl.col("model")==model)&(pl.col("task")==t)&(pl.col("regime")==regime))
                if r.height==0: continue
                r = r.row(0, named=True)
                print(f"  {model:8s} {regime:9s} gold={r['gold_f1']:.4f}±{r['gold_f1_err']:.4f} "
                      f"bench={r['bench_f1']:.4f}±{r['bench_f1_err']:.4f} (bench_n={r['bench_n']})")

    # 2. Few-shot vs zero-shot gain per teacher per task (gold + bench)
    print("\n--- FEWSHOT − ZEROSHOT gain per teacher ---")
    for t in TASKS:
        for model in ["mistral","gemma","sonnet"]:
            fs = df.filter((pl.col("model")==model)&(pl.col("task")==t)&(pl.col("regime")=="fewshot"))
            zs = df.filter((pl.col("model")==model)&(pl.col("task")==t)&(pl.col("regime")=="zeroshot"))
            if fs.height==0 or zs.height==0: continue
            fs, zs = fs.row(0,named=True), zs.row(0,named=True)
            print(f"  {t:12s} {model:8s} Δgold={fs['gold_f1']-zs['gold_f1']:+.4f} "
                  f"Δbench={fs['bench_f1']-zs['bench_f1']:+.4f}")

    # 3. Best teacher (any model/regime) vs the student, per task per eval
    print("\n--- BEST TEACHER vs STUDENT ---")
    for t in TASKS:
        for eval_, col in [("gold","gold_f1"),("bench","bench_f1")]:
            teach = df.filter((pl.col("task")==t)&(pl.col("model")!="student"))
            stud  = df.filter((pl.col("task")==t)&(pl.col("model")=="student"))
            if teach.height==0 or stud.height==0: continue
            bt = teach.sort(col, descending=True).row(0, named=True)
            st = stud.row(0, named=True)
            print(f"  {t:12s} {eval_:5s} best_teacher={bt['model']}/{bt['regime']} "
                  f"{bt[col]:.4f} | student {st[col]:.4f} | Δ(stud−teach)={st[col]-bt[col]:+.4f}")

    # 4. Sonnet (arbiter) as a rough label-quality ceiling, few-shot
    print("\n--- SONNET few-shot (arbiter/annotation ceiling) ---")
    for t in TASKS:
        r = df.filter((pl.col("model")=="sonnet")&(pl.col("task")==t)&(pl.col("regime")=="fewshot"))
        if r.height==0: continue
        r = r.row(0, named=True)
        print(f"  {t:12s} gold={r['gold_f1']:.4f} bench={r['bench_f1']:.4f} (bench_n={r['bench_n']})")

if __name__ == "__main__":
    df = main()
    print_teacher_stats(df)
