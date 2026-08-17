"""
OOD baseline contextualisation for all four tasks.
Computes majority/random floors, gold-supervised vs. student per-class F1,
collapse counts, and a paired bootstrap CI on the OOD macro-F1 gap.
"""
import json, glob
import numpy as np

# ------------------------------------------------------------------ CONFIG
BASE = "/"

TASKS = {
    "hate": {
        "student_glob": f"{BASE}/results/stage5_results by fold/"
                        f"hate_twitter_128000_fold*.json",
        "gold_glob":    f"{BASE}/results/stage6_human_supervised_bundle/"
                        f"stage6_human_supervised_results/hate_human_supervised_fold*.json",
        "class_names": {0: "hate", 1: "offensive", 2: "neither"},
        "n_classes": 3,
    },
    "sexism": {
        "student_glob": f"{BASE}/results/stage5_results by fold/"
                        f"sexism_twitter_128000_fold*.json",
        "gold_glob":    f"{BASE}/results/stage6_human_supervised_bundle/"
                        f"stage6_human_supervised_results/sexism_human_supervised_fold*.json",
        "class_names": {0: "not_sexist", 1: "sexist"},
        "n_classes": 2,
    },
    "lexical_bias": {
        "student_glob": f"{BASE}/results/stage5_results by fold/"
                        f"lexical_bias_news_128000_fold*.json",
        "gold_glob":    f"{BASE}/results/stage6_human_supervised_bundle/"
                        f"stage6_human_supervised_results/lexical_bias_human_supervised_fold*.json",
        "class_names": {0: "not_biased", 1: "biased"},
        "n_classes": 2,
    },
    "sentiment": {
        "student_glob": f"{BASE}/results/stage5_results by fold/"
                        f"sentiment_news_128000_fold*.json",
        "gold_glob":    f"{BASE}/results/stage6_human_supervised_bundle/"
                        f"stage6_human_supervised_results/sentiment_human_supervised_fold*.json",
        "class_names": {0: "negative", 1: "neutral", 2: "positive"},
        "n_classes": 3,
    },
}

N_BOOT   = 10000
RNG_SEED = 42

# ------------------------------------------------------------------ LOADING
def load_folds(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"[{label}] No files matched:\n  {pattern}\n"
            f"Check the directory name (spaces are OK in Python globs) "
            f"and the fold-file prefix."
        )
    print(f"[{label}] matched {len(files)} file(s):")
    for f in files:
        print("   ", f)
    return [json.load(open(f)) for f in files]

# ------------------------------------------------------------------ HELPERS
def counts_from_dist(dist, n_classes):
    """Accept a *_dist field as list [c0,...] or dict {'0':c0,...}/{0:c0,...}."""
    if isinstance(dist, dict):
        return np.array([float(dist[str(k)] if str(k) in dist else dist[k])
                         for k in range(n_classes)])
    return np.array([float(dist[k]) for k in range(n_classes)])

def majority_floor(true_dist, n_classes):
    """Analytic macro-F1 of an 'always predict majority class' classifier."""
    c = counts_from_dist(true_dist, n_classes)
    p = c / c.sum(); m = int(c.argmax())
    f1 = np.zeros(n_classes)
    f1[m] = 2 * p[m] / (1 + p[m])
    return f1, float(f1.mean()), float(p[m]), m

def random_floor(true_dist, n_classes):
    """Expected macro-F1 of a stratified-random classifier (E[F1_i] = p_i)."""
    c = counts_from_dist(true_dist, n_classes)
    p = c / c.sum()
    return p, float(p.mean())

def per_class_f1_array(folds, split, i):
    return np.array([f[f"{split}_f1_class_{i}"] for f in folds])

def aggregate(folds, split, n_classes):
    macro = np.array([f[f"{split}_macro_f1"]  for f in folds])
    acc   = np.array([f[f"{split}_accuracy"]  for f in folds])
    pc    = {i: per_class_f1_array(folds, split, i) for i in range(n_classes)}
    coll  = sum(int(f.get(f"{split}_collapsed", False)) for f in folds)
    return {"macro": macro, "acc": acc, "pc": pc,
            "n_collapsed": coll, "n_folds": len(folds)}

def fmt(a):  # mean±sd for an array
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"

# ------------------------------------------------------------------ PAIRED BOOTSTRAP
def paired_bootstrap_ci(student_macro, gold_macro, n_boot=N_BOOT, seed=RNG_SEED):
    """
    Fold-paired bootstrap on the student-minus-gold macro-F1 gap.
    Folds are seed-matched across the two file sets, so we resample fold indices
    jointly. Returns (mean_gap, lo95, hi95, p_two_sided).
    """
    assert len(student_macro) == len(gold_macro), \
        "Student and gold must have the same number of (seed-matched) folds."
    rng = np.random.default_rng(seed)
    diffs = student_macro - gold_macro
    n = len(diffs)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)     # resample folds with replacement
        boot[b] = diffs[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided p: proportion of bootstrap means on the opposite side of 0
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(min(p, 1.0))

# ------------------------------------------------------------------ PER-TASK RUN
def run_task(task, cfg):
    n_classes   = cfg["n_classes"]
    class_names = cfg["class_names"]

    print("\n" + "#" * 64)
    print(f"# TASK: {task}")
    print("#" * 64)

    student_folds = load_folds(cfg["student_glob"], f"{task} STUDENT")
    gold_folds    = load_folds(cfg["gold_glob"],    f"{task} GOLD/human_supervised")

    # Benchmark label distribution is fixed across conditions -> take from gold fold 0
    bench_true = gold_folds[0]["bench_true_dist"]
    print("\nbench_true_dist type:", type(bench_true).__name__, "| value:", bench_true)

    maj_f1, maj_macro, maj_acc, maj_idx = majority_floor(bench_true, n_classes)
    rand_f1, rand_macro                  = random_floor(bench_true, n_classes)
    shares = counts_from_dist(bench_true, n_classes)
    shares = shares / shares.sum()

    stu = aggregate(student_folds, "bench", n_classes)
    gld = aggregate(gold_folds,    "bench", n_classes)

    gap_mean, gap_lo, gap_hi, gap_p = paired_bootstrap_ci(stu["macro"], gld["macro"])

    # -------------------------------------------------------------- REPORT
    print("\n" + "=" * 64)
    print(f"OOD BENCHMARK — {task}")
    print("=" * 64)
    print("Class shares: " +
          "  ".join(f"{class_names[i]}={shares[i]:.3f}" for i in range(n_classes)))

    print(f"\nMajority ('{class_names[maj_idx]}')  "
          f"acc={maj_acc:.3f}  macro-F1={maj_macro:.3f}  "
          f"per-class F1={ {class_names[i]: round(maj_f1[i],3) for i in range(n_classes)} }")
    print(f"Stratified-random           macro-F1={rand_macro:.3f}  "
          f"per-class F1={ {class_names[i]: round(rand_f1[i],3) for i in range(n_classes)} }")

    print(f"\nGold-supervised  macro-F1={fmt(gld['macro'])}  acc={fmt(gld['acc'])}  "
          f"collapsed={gld['n_collapsed']}/{gld['n_folds']}")
    print("  per-class F1:", {class_names[i]: fmt(gld["pc"][i]) for i in range(n_classes)})

    print(f"\nStudent (best)   macro-F1={fmt(stu['macro'])}  acc={fmt(stu['acc'])}  "
          f"collapsed={stu['n_collapsed']}/{stu['n_folds']}")
    print("  per-class F1:", {class_names[i]: fmt(stu["pc"][i]) for i in range(n_classes)})

    print(f"\nStudent - Gold OOD macro-F1 gap = {gap_mean:+.3f}  "
          f"95% CI [{gap_lo:+.3f}, {gap_hi:+.3f}]  bootstrap p={gap_p:.4f}")
    print(f"Gold above majority floor:    {gld['macro'].mean() - maj_macro:+.3f}")
    print(f"Student above majority floor: {stu['macro'].mean() - maj_macro:+.3f}")

    return {
        "task": task,
        "student": stu["macro"].mean(),
        "gold": gld["macro"].mean(),
        "gap": gap_mean, "lo": gap_lo, "hi": gap_hi, "p": gap_p,
    }

# ------------------------------------------------------------------ RUN ALL
summary = []
for task, cfg in TASKS.items():
    try:
        summary.append(run_task(task, cfg))
    except FileNotFoundError as e:
        print(f"\n[SKIP] {task}: {e}")

# ------------------------------------------------------------------ SUMMARY TABLE
print("\n" + "=" * 78)
print("SUMMARY — student vs. gold OOD macro-F1 (paired bootstrap)")
print("=" * 78)
print(f"{'task':<14}{'student':>9}{'gold':>8}{'gap':>9}"
      f"{'95% CI':>20}{'p':>10}{'sig?':>7}")
for r in summary:
    ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
    sig = "yes" if r["p"] < 0.05 else "no"
    print(f"{r['task']:<14}{r['student']:>9.3f}{r['gold']:>8.3f}"
          f"{r['gap']:>+9.3f}{ci:>20}{r['p']:>10.4f}{sig:>7}")
