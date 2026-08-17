"""Stage 6 — Baseline LLM evaluation (B1 Mistral, B2 Gemma, B3 Sonnet 5).

LLM arm. Evaluates the three annotator
LLMs.

Two prompt regimes, run in one invocation via --regime both:
  - "fewshot" : identical 8-shot retrieval prompt used during annotation
                (all-mpnet-base-v2, K_SHOTS=8, shots drawn from *_gold_train,
                 disjoint from *_gold_eval by the seeded 80/20 split, and a
                 different corpus from the benchmarks -> no leakage).
  - "zeroshot": same prompt template, zero retrieved examples.

Eval sets per task (both use text='text', label='label_str', canonical):
  - gold      : held-out 20% human gold (*_gold_eval.parquet), in-domain
  - benchmark : external OOD resource (*_benchmark.parquet)

Metrics per (model x task x eval_set x regime):
  - macro-F1 and MCC, reported TWO ways:
      * honest  : nulls/refusals mapped to a sentinel -> counted as errors
      * covcond : coverage-conditional, computed on parsed rows only
  - bootstrap 95% CIs (2000 eval-set resamples) so single-pass LLMs compare
    fairly against the student's 6-fold spread.
  - coverage / n_null / api_errors logged explicitly (no silent null drops).

Run (no-GPU first per priority):
  python stage6_baseline_llm.py --model mistral --regime both --eval both --task all
  python stage6_baseline_llm.py --model gemma   --regime both --eval both --task all
  # Sonnet via Anthropic Batch:
  python stage6_baseline_llm.py --model sonnet --regime both --eval both --task all --submit
  python stage6_baseline_llm.py --model sonnet --regime both --eval both --task all --collect
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score, matthews_corrcoef

# Twitter config defines ALL FOUR tasks (definition + labels).
from stage4_config import (TASKS, TEACHERS, EMBED_MODEL, INDEX_DIR,
                           DEEPINFRA_BASE_URL, K_SHOTS)
import warnings
from sklearn.exceptions import UndefinedMetricWarning


warnings.filterwarnings("ignore", message="A single label was found",
                        category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BENCH_DIR = Path.home() / "/data/stage5_benchmarks"
RESULTS_DIR = Path.home() / "/results/stage6_baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SONNET_BATCH_DIR = RESULTS_DIR / "sonnet_batches"
SONNET_BATCH_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RESULTS_DIR / "stage6_baseline_summary.csv"

# --------------------------------------------------------------------------
# Eval-set config
#   * text column is "text" everywhere (gold_eval + benchmark, all tasks)
#   * label column is "label_str" everywhere, already canonical
#   * id may be absent (hate/sentiment gold_eval)
# --------------------------------------------------------------------------
BENCH_FILE = {
    "hate":         "hate_benchmark.parquet",
    "sexism":       "sexism_benchmark.parquet",
    "sentiment":    "sentiment_benchmark.parquet",
    "lexical_bias": "lexical_bias_benchmark.parquet",
}
TEXT_COL = "text"
LABEL_COL = "label_str"

SONNET_MODEL = "claude-sonnet-5"
SYSTEM = ("You are a classification model. Respond with exactly one label "
          "and nothing else. No preamble, no explanation, no analysis.")
.
MODEL_PARAMS = {
    "mistral": "24B",
    "gemma":   "27B",
    "sonnet":  "frontier (undisclosed)",
}
MODEL_ROLE = {
    "mistral": "full teacher (100% of labels)",
    "gemma":   "full teacher (100% of labels)",
    "sonnet":  "partial teacher (~11% tiebreaker)",
}

N_BOOT = 2000
BOOT_SEED = 42


def build_prompt(cfg, examples, target):
    lines = [f"You are an expert in {cfg['task_name']} classification.",
             f"{cfg['task_name'].capitalize()} is {cfg['definition']}",
             f"Your task is to classify each sentence as one of: "
             f"{', '.join(cfg['labels'])}.",
             f"Respond with only the label, no explanation, no preamble."]
    for txt, lab in examples:
        lines.append(f"Sentence: {txt}")
        lines.append(f"Label: {lab}")
    lines.append(f"Sentence: {target}")
    lines.append("Label:")
    return "\n".join(lines)

BENCH_CAP_SEED = 1234

def stratified_cap(df, label_col, cap, seed=BENCH_CAP_SEED):
    """Return df capped to ~`cap` rows, preserving per-class proportions.
    Deterministic given seed. If df.height <= cap, returns df unchanged."""
    n = df.height
    if cap is None or n <= cap:
        return df
    frac = cap / n
    parts = []
    for lab, grp in df.group_by(label_col, maintain_order=True):
        # at least 1 per present class; round the rest
        k = max(1, int(round(grp.height * frac)))
        k = min(k, grp.height)
        parts.append(grp.sample(n=k, seed=seed, shuffle=True))
    out = pl.concat(parts)
    # trim/pad to exactly cap only if you want a hard size; proportions kept as-is
    return out

def parse_label(raw, task):
    """Task-specific label extraction. Returns None on null/unparseable."""
    if raw is None:
        return None
    up = raw.upper().strip()

    if task == "sexism":
        if "NOT" in up and ("SEX" in up or "SEXI" in up):
            return "NOT SEXIST"
        if "SEXIST" in up or "SEXI" in up or up.startswith("SEX"):
            return "SEXIST"
        return None

    if task == "lexical_bias":
        if "NOT" in up and "BIAS" in up:
            return "NOT BIASED"
        if "BIAS" in up:
            return "BIASED"
        return None

    if task == "hate":
        if "NEITHER" in up or "NONE" in up:
            return "NEITHER"
        if "HATE" in up:
            return "HATE SPEECH"
        if "OFFENSIVE" in up or "OFFEN" in up:
            return "OFFENSIVE"
        return None

    if task == "sentiment":
        if "POSITIVE" in up or up.startswith("POS"):
            return "POSITIVE"
        if "NEGATIVE" in up or up.startswith("NEG"):
            return "NEGATIVE"
        if "NEUTRAL" in up or up.startswith("NEU"):
            return "NEUTRAL"
        return None

    return None


def norm(s):
    """Canonicalize label strings for safe equality (whitespace/case)."""
    return None if s is None else str(s).strip().upper()


# --------------------------------------------------------------------------
# Eval-set loading (gold_eval + benchmark share identical column logic)
# --------------------------------------------------------------------------
def load_eval(task, which, bench_cap=None):
    """Return (ids, texts, golds_canonical) for 'gold' or 'benchmark'."""
    if which == "gold":
        path = INDEX_DIR / f"{task}_gold_eval.parquet"
    elif which == "benchmark":
        path = BENCH_DIR / BENCH_FILE[task]
    else:
        raise ValueError(which)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pl.read_parquet(path)
    for col in (TEXT_COL, LABEL_COL):
        if col not in df.columns:
            raise KeyError(f"{path} missing column '{col}' "
                           f"(has {df.columns})")

    df = df.filter(pl.col(TEXT_COL).is_not_null()
                   & pl.col(LABEL_COL).is_not_null())
    if which == "benchmark" and bench_cap:
        before = df.height
        df = stratified_cap(df, LABEL_COL, bench_cap)
        print(f"  [{task}/benchmark] stratified cap {before:,} -> {df.height:,} "
              f"(seed={BENCH_CAP_SEED})")
        print("   class balance:",
              df.group_by(LABEL_COL).agg(pl.len()).sort(LABEL_COL).to_dicts())

    # id may be absent (hate/sentiment gold_eval) -> positional fallback
    if "id" in df.columns:
        ids = [str(x) for x in df["id"].to_list()]
    else:
        ids = [str(i) for i in range(df.height)]

    texts = df[TEXT_COL].to_list()
    golds = [norm(g) for g in df[LABEL_COL].to_list()]
    # --- test-only row cap (no effect unless STAGE6_LIMIT is set) ---
    _lim = os.environ.get("STAGE6_LIMIT")
    if _lim:
        ids, texts, golds = ids[:int(_lim)], texts[:int(_lim)], golds[:int(_lim)]
    return ids, texts, golds


def build_shots(task, texts, regime):
    """Per-target list of (example_txt, example_lab). Empty for zeroshot.
    Retrieval identical to stage4a/stage4b: mpnet embeddings, cosine top-K
    from *_gold_train (disjoint from *_gold_eval; different corpus from bench)."""
    if regime == "zeroshot":
        return [[] for _ in texts]

    from sentence_transformers import SentenceTransformer
    cfg = TASKS[task]
    train = pl.read_parquet(INDEX_DIR / f"{task}_gold_train.parquet")
    train_txt = train[cfg["gold_text_col"]].to_list()
    train_lab = train["label_str"].to_list()
    train_emb = np.load(INDEX_DIR / f"{task}_train_emb.npy")

    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"[{task}/{regime}] embedding {len(texts):,} targets for retrieval...")
    q_emb = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True).astype(np.float32)
    shots = []
    for j in range(len(texts)):
        sims = train_emb @ q_emb[j]
        top = np.argsort(-sims)[:K_SHOTS]
        shots.append([(train_txt[k], train_lab[k]) for k in top])
    return shots


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
_NULL = "__NULL__"


def _f1(a, b, labels):
    return f1_score(a, b, labels=labels, average="macro", zero_division=0)


def _mcc(a, b):
    return matthews_corrcoef(a, b)


def compute_metrics(y_true, y_pred, labels):
    """y_pred may contain None. Report metrics two ways + coverage/nulls."""
    yt = [norm(t) for t in y_true]
    yp = [norm(p) if p is not None else None for p in y_pred]
    n = len(yt)
    n_null = sum(1 for p in yp if p is None)

    yp_honest = [p if p is not None else _NULL for p in yp]
    honest_f1 = _f1(yt, yp_honest, labels)
    honest_mcc = _mcc(yt, yp_honest)

    keep = [i for i in range(n) if yp[i] is not None]
    if keep:
        yt_c = [yt[i] for i in keep]
        yp_c = [yp[i] for i in keep]
        cov_f1 = _f1(yt_c, yp_c, labels)
        cov_mcc = _mcc(yt_c, yp_c)
    else:
        yt_c, yp_c = [], []
        cov_f1 = cov_mcc = 0.0

    # --- bootstrap CIs (resample eval set) ---
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.arange(n)
    keep_arr = np.array(keep, dtype=int)
    hf, hm, cf, cm = [], [], [], []
    for _ in range(N_BOOT):
        s = rng.choice(idx, size=n, replace=True)
        a = [yt[i] for i in s]
        b = [yp_honest[i] for i in s]
        hf.append(_f1(a, b, labels))
        hm.append(_mcc(a, b))
        # coverage-conditional bootstrap: resample only parsed indices
        if keep_arr.size:
            s2 = rng.choice(keep_arr, size=keep_arr.size, replace=True)
            a2 = [yt[i] for i in s2]
            b2 = [yp[i] for i in s2]
            cf.append(_f1(a2, b2, labels))
            cm.append(_mcc(a2, b2))
        else:
            cf.append(0.0)
            cm.append(0.0)

    def ci(arr):
        return (float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)))

    return {
        "n": n,
        "n_null": n_null,
        "coverage": (n - n_null) / n if n else 0.0,
        # honest
        "honest_macro_f1": honest_f1, "honest_macro_f1_ci": ci(hf),
        "honest_mcc": honest_mcc, "honest_mcc_ci": ci(hm),
        # coverage-conditional
        "cov_macro_f1": cov_f1, "cov_macro_f1_ci": ci(cf),
        "cov_mcc": cov_mcc, "cov_mcc_ci": ci(cm),
    }


# --------------------------------------------------------------------------
# DeepInfra path (Mistral / Gemma)
# --------------------------------------------------------------------------
def run_deepinfra(model_key, task, which, regime, workers=32):
    from openai import OpenAI
    model_id = TEACHERS[model_key]
    api_key = os.environ.get("DEEPINFRA_TOKEN")
    if not api_key:
        raise SystemExit("Set DEEPINFRA_TOKEN in the environment.")
    client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
    cfg = TASKS[task]

    ids, texts, golds = load_eval(task, which)
    shots = build_shots(task, texts, regime)
    prompts = [build_prompt(cfg, shots[j], texts[j]) for j in range(len(texts))]

    def one(prompt):
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    max_tokens=8, temperature=0.0, timeout=60)
                return r.choices[0].message.content.strip(), None
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    return None, str(e)
                time.sleep(2 ** attempt)
        return None, "max_retries_exceeded"

    raws = [None] * len(prompts)
    errs = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, prompts[i]): i for i in range(len(prompts))}
        for done, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            raw, err = fut.result()
            if err:
                errs += 1
            raws[i] = raw
            if done % 1000 == 0 or done == len(prompts):
                el = time.time() - t0
                print(f"  [{model_key}/{task}/{which}/{regime}] "
                      f"{done:,}/{len(prompts):,} | {done/el:.1f} inst/s | "
                      f"errs {errs}", flush=True)

    preds = [parse_label(r, task) for r in raws]
    return ids, texts, golds, preds, raws, errs


# --------------------------------------------------------------------------
# Anthropic Batch path (Sonnet 5) — submit / collect, matches stage4c
# --------------------------------------------------------------------------
def sonnet_submit(task, which, regime, bench_cap=None):
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY in the environment.")
    client = Anthropic(api_key=api_key)
    cfg = TASKS[task]

    ids, texts, golds = load_eval(task, which, bench_cap=bench_cap)
    shots = build_shots(task, texts, regime)

    # custom_id must be unique; we use the POSITION j and keep aligned arrays
    # on disk so id-dtype/duplication never breaks reassembly.
    requests = []
    for j in range(len(texts)):
        prompt = build_prompt(cfg, shots[j], texts[j])
        requests.append({
            "custom_id": str(j),
            "params": {"model": SONNET_MODEL, "max_tokens": 8,
                       "system": SYSTEM,
                       "messages": [{"role": "user", "content": prompt}]},
        })

    batch = client.messages.batches.create(requests=requests)
    tag = f"sonnet_{task}_{which}_{regime}"
    meta = {"batch_id": batch.id, "bench_cap": bench_cap, "task": task, "which": which,
            "regime": regime, "n": len(requests),
            "ids": ids, "texts": texts, "golds": golds}
    (SONNET_BATCH_DIR / f"{tag}.json").write_text(json.dumps(meta))
    print(f"[{tag}] submitted batch {batch.id} ({len(requests):,} reqs)")


def sonnet_collect(task, which, regime, poll=30):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tag = f"sonnet_{task}_{which}_{regime}"
    meta_path = SONNET_BATCH_DIR / f"{tag}.json"
    if not meta_path.exists():
        raise SystemExit(f"No batch metadata for {tag} — run --submit first.")
    meta = json.loads(meta_path.read_text())
    bid, n = meta["batch_id"], meta["n"]

    while True:
        b = client.messages.batches.retrieve(bid)
        c = b.request_counts
        print(f"[{tag}] {b.processing_status} done={c.succeeded} "
              f"err={c.errored} proc={c.processing}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(poll)

    raws = [None] * n
    for res in client.messages.batches.results(bid):
        j = int(res.custom_id)  # positional
        if res.result.type == "succeeded":
            raw = None
            for blk in res.result.message.content:  # skip thinking blocks
                if getattr(blk, "type", None) == "text":
                    raw = blk.text.strip()
                    break
            raws[j] = raw if raw else None
        else:
            raws[j] = None  # errored/expired/canceled -> explicit null

    preds = [parse_label(r, task) for r in raws]
    return meta["ids"], meta["texts"], meta["golds"], preds, raws, 0


# --------------------------------------------------------------------------
# Save per-row audit dump + append a summary row
# --------------------------------------------------------------------------
def save_and_score(model_key, task, which, regime,
                   ids, texts, golds, preds, raws, errs):
    cfg = TASKS[task]
    labels = [norm(l) for l in cfg["labels"]]
    tag = f"{model_key}_{task}_{which}_{regime}"

    # per-row audit (id, text, gold, pred, raw)
    pl.DataFrame({
        "id": [str(i) for i in ids],
        "text": texts,
        "gold": [norm(g) for g in golds],
        "pred": [norm(p) if p is not None else None for p in preds],
        "raw": [r if r is not None else None for r in raws],
    }).write_parquet(RESULTS_DIR / f"{tag}_preds.parquet")

    m = compute_metrics(golds, preds, labels)

    row = {
        "model": model_key,
        "role": MODEL_ROLE.get(model_key, ""),
        "params": MODEL_PARAMS.get(model_key, ""),
        "task": task,
        "eval_set": which,
        "regime": regime,
        "api_errors": errs,
        "n": m["n"], "n_null": m["n_null"], "coverage": m["coverage"],
        "honest_macro_f1": m["honest_macro_f1"],
        "honest_macro_f1_lo": m["honest_macro_f1_ci"][0],
        "honest_macro_f1_hi": m["honest_macro_f1_ci"][1],
        "honest_mcc": m["honest_mcc"],
        "honest_mcc_lo": m["honest_mcc_ci"][0],
        "honest_mcc_hi": m["honest_mcc_ci"][1],
        "cov_macro_f1": m["cov_macro_f1"],
        "cov_macro_f1_lo": m["cov_macro_f1_ci"][0],
        "cov_macro_f1_hi": m["cov_macro_f1_ci"][1],
        "cov_mcc": m["cov_mcc"],
        "cov_mcc_lo": m["cov_mcc_ci"][0],
        "cov_mcc_hi": m["cov_mcc_ci"][1],
    }

    dfrow = pl.DataFrame([row])
    if SUMMARY_PATH.exists():
        old = pl.read_csv(SUMMARY_PATH)
        # de-dup on the identifying keys so re-runs overwrite cleanly
        keyset = {(r["model"], r["task"], r["eval_set"], r["regime"])
                  for r in [row]}
        old = old.filter(
            ~pl.struct(["model", "task", "eval_set", "regime"]).map_elements(
                lambda s: (s["model"], s["task"], s["eval_set"], s["regime"]) in keyset,
                return_dtype=pl.Boolean))
        pl.concat([old, dfrow], how="diagonal_relaxed").write_csv(SUMMARY_PATH)
    else:
        dfrow.write_csv(SUMMARY_PATH)

    print(f"[{tag}] honest: F1={m['honest_macro_f1']:.3f} "
          f"[{m['honest_macro_f1_ci'][0]:.3f},{m['honest_macro_f1_ci'][1]:.3f}] "
          f"MCC={m['honest_mcc']:.3f} "
          f"[{m['honest_mcc_ci'][0]:.3f},{m['honest_mcc_ci'][1]:.3f}]")
    print(f"[{tag}] cov   : F1={m['cov_macro_f1']:.3f} "
          f"[{m['cov_macro_f1_ci'][0]:.3f},{m['cov_macro_f1_ci'][1]:.3f}] "
          f"MCC={m['cov_mcc']:.3f} "
          f"[{m['cov_mcc_ci'][0]:.3f},{m['cov_mcc_ci'][1]:.3f}]")
    print(f"[{tag}] null={m['n_null']}/{m['n']} "
          f"cov={m['coverage']:.3f} apierr={errs}")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mistral", "gemma", "sonnet"])
    ap.add_argument("--task", default="all",
                    help="'all' or one of: " + ", ".join(TASKS))
    ap.add_argument("--regime", default="both",
                    choices=["fewshot", "zeroshot", "both"])
    ap.add_argument("--eval", default="both",
                    choices=["gold", "benchmark", "both"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--submit", action="store_true",
                    help="sonnet: create + submit Anthropic batches")
    ap.add_argument("--collect", action="store_true",
                    help="sonnet: poll + download + score submitted batches")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--bench-cap", type=int, default=None,
                    help="stratified row cap on benchmark eval only "
                         "(gold stays full). e.g. 5000")
    args = ap.parse_args()

    if args.task != "all" and args.task not in TASKS:
        raise SystemExit(f"Unknown task '{args.task}'. Choices: all, "
                         + ", ".join(TASKS))

    tasks = list(TASKS) if args.task == "all" else [args.task]
    regimes = (["fewshot", "zeroshot"] if args.regime == "both"
               else [args.regime])
    evals = (["gold", "benchmark"] if args.eval == "both"
             else [args.eval])

    print("=" * 70)
    print(f"Stage 6 baseline LLM eval | model={args.model} "
          f"role={MODEL_ROLE.get(args.model, '?')}")
    print(f"tasks={tasks}")
    print(f"regimes={regimes} eval_sets={evals}")
    if args.model == "sonnet":
        mode = "SUBMIT" if args.submit else "COLLECT" if args.collect else "?"
        print(f"sonnet mode={mode}")
    print("=" * 70)

    for task in tasks:
        for which in evals:
            for regime in regimes:
                print(f"\n>>> {args.model} | {task} | {which} | {regime}")
                if args.model in ("mistral", "gemma"):
                    out = run_deepinfra(args.model, task, which, regime,
                                        workers=args.workers)
                    save_and_score(args.model, task, which, regime, *out)
                else:  # sonnet
                    if args.submit:
                        sonnet_submit(task, which, regime, bench_cap=args.bench_cap)
                    elif args.collect:
                        out = sonnet_collect(task, which, regime, poll=args.poll)
                        save_and_score("sonnet", task, which, regime, *out)
                    else:
                        raise SystemExit(
                            "sonnet requires --submit or --collect")

    print("\n" + "=" * 70)
    print(f"Done. Summary appended to: {SUMMARY_PATH}")
    print("Per-row audit dumps in:", RESULTS_DIR)
    print("=" * 70)

if __name__ == "__main__":
    main()

