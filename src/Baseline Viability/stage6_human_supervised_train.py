"""Stage 6 — B4 Human-Supervised human_supervised.

train on the HUMAN gold_train partition instead of synthetic-label
scaling pools. Everything else (model, MAX_LEN, class-weighted CE,
warmup 0.06, grad-clip 1.0, bf16, per-fold seed offsets, collapse
detection, per-fold JSON schema) is preserved so B4 is the student's
recipe with human labels swapped in.

Differences from stage5b:
  * No size loop, no source loop — one human_supervised per task on full gold_train.
  * 6-fold split built ON gold_train (stratified on final_label), each
    fold-model predicts the FIXED gold_eval + FULL benchmark.
  * Added matthews_corrcoef (MCC) to metrics (B4 column needs macro-F1 AND MCC).
  * Runtime disjointness assert: gold_train vs gold_eval (id for sexism,
    text otherwise).

Usage:
  python stage6_human_supervised_train.py --task sexism --fold 0
"""
import argparse, json, shutil, time
import numpy as np, polars as pl
import torch
from torch import nn
from datasets import Dataset
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, DataCollatorWithPadding,
                          Trainer, set_seed)

from stage5_config import (TASKS, STUDENT_MODEL, HYPERPARAMS, MAX_LEN,
                           BATCH_SIZE, SEED, N_FOLDS, LABEL2ID, ID2LABEL, DATA)


INDEX_DIR   = DATA / "stage4_index"
BENCH_DIR   = DATA / "stage5_benchmarks"
human_supervised_DIR  = DATA / "stage6_human_supervised_results"
human_supervised_DIR.mkdir(parents=True, exist_ok=True)


WARMUP_RATIO      = 0.06
MAX_GRAD_NORM     = 1.0
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING   = 0.0


# ----------------------------------------------------------------------
# Data loading — gold_train / gold_eval / benchmark
# ----------------------------------------------------------------------
def _load_gold(path, task):
    """Load a gold parquet (text + label_str) -> text_clean/final_label."""
    df = pl.read_parquet(path)
    df = df.filter(pl.col("text").is_not_null()
                   & pl.col("label_str").is_not_null())
    return df.rename({"text": "text_clean", "label_str": "final_label"})


def load_gold_train(task):
    p = INDEX_DIR / f"{task}_gold_train.parquet"
    if not p.exists():
        raise SystemExit(f"Missing gold_train file {p}")
    keep_id = "id" if "id" in pl.read_parquet(p).columns else None
    df = _load_gold(p, task)
    return df, keep_id


def load_gold_eval(task):
    p = INDEX_DIR / f"{task}_gold_eval.parquet"
    return _load_gold(p, task)


def load_benchmark(task):
    p = BENCH_DIR / f"{task}_benchmark.parquet"
    if not p.exists():
        print(f"  [warn] no benchmark file for {task} at {p}; skipping OOD eval")
        return None
    df = pl.read_parquet(p)
    df = df.filter(pl.col("text").is_not_null()
                   & pl.col("label_str").is_not_null())
    return df.rename({"label_str": "final_label", "text": "text_clean"})


def assert_disjoint(task):
    """Leakage guard: gold_train must not overlap gold_eval.
    sexism has 'id' -> check ids AND text; others -> check text (stronger
    than positional, and the only shared key available)."""
    tr = pl.read_parquet(INDEX_DIR / f"{task}_gold_train.parquet")
    ev = pl.read_parquet(INDEX_DIR / f"{task}_gold_eval.parquet")
    if "id" in tr.columns and "id" in ev.columns:
        overlap_id = set(tr["id"].to_list()) & set(ev["id"].to_list())
        print(f"  [disjoint] {task}: id-overlap = {len(overlap_id)}")
        assert not overlap_id, f"{task}: {len(overlap_id)} shared ids train/eval!"
    overlap_txt = set(tr["text"].to_list()) & set(ev["text"].to_list())
    print(f"  [disjoint] {task}: text-overlap = {len(overlap_txt)} "
          f"(train={tr.height}, eval={ev.height})")
    if overlap_txt:
        print(f"  [!! WARN] {task}: {len(overlap_txt)} texts appear in BOTH "
              f"train and eval — investigate before trusting B4.")


# ----------------------------------------------------------------------
# 6-fold split on gold_train (stratified on final_label)
# ----------------------------------------------------------------------
def assign_folds(df, n_folds, seed, strat_col="final_label"):
    keys = df[strat_col].to_numpy()
    idx = np.arange(len(keys))
    fold_col = np.full(len(keys), -1, dtype=np.int64)
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(skf.split(idx, keys)):
            fold_col[te] = k
    except ValueError as e:
        print(f"    stratified split failed ({e}); using plain KFold")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(kf.split(idx)):
            fold_col[te] = k
    return df.with_columns(pl.Series("fold", fold_col))


def to_hf(df, task, tokenizer, split_name=""):
    l2i = LABEL2ID[task]
    before = df.height
    df = df.filter(pl.col("final_label").is_in(list(l2i.keys())))
    dropped = before - df.height
    if dropped:
        print(f"  [warn] {split_name}: dropped {dropped}/{before} rows with "
              f"labels outside {set(l2i)} (kept {df.height})")
    if df.height == 0:
        raise SystemExit(f"[{task}] 0 rows after label filter in '{split_name}'")
    texts = df["text_clean"].to_list()
    labels = [l2i[x] for x in df["final_label"].to_list()]
    ds = Dataset.from_dict({"text": texts, "label": labels})
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True,
                                    max_length=MAX_LEN),
                batched=True, remove_columns=["text"])
    return ds, np.array(labels)


def class_dist(y, n):
    return np.bincount(y, minlength=n).tolist()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = (self.class_weights.to(logits.device)
             if self.class_weights is not None else None)
        loss_fct = nn.CrossEntropyLoss(weight=w,
                                       label_smoothing=self.label_smoothing)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def make_metrics_fn(num_labels):
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0)
        f1_per = f1_score(labels, preds, average=None,
                          labels=list(range(num_labels)), zero_division=0)
        out = {
            "macro_f1": f1, "macro_precision": p, "macro_recall": r,
            "accuracy": accuracy_score(labels, preds),
            "mcc": matthews_corrcoef(labels, preds),   # ADDED for B4 column
        }
        for i, v in enumerate(f1_per):
            out[f"f1_class_{i}"] = float(v)
        out["n_pred_classes"] = int(len(np.unique(preds)))
        return out
    return compute


def eval_with_diag(trainer, ds, y_true, num_labels, prefix, id2label):
    pred = trainer.predict(ds)
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pd_ = class_dist(pred_ids, num_labels)
    td_ = class_dist(y_true, num_labels)
    names = [id2label[i] for i in range(num_labels)]
    print(f"  [{prefix}] pred dist: "
          + ", ".join(f"{names[i]}={pd_[i]}" for i in range(num_labels)))
    print(f"  [{prefix}] true dist: "
          + ", ".join(f"{names[i]}={td_[i]}" for i in range(num_labels)))
    collapsed = len(set(pred_ids)) == 1
    if collapsed:
        print(f"  [COLLAPSE] {prefix}: predicts ONLY '{names[pred_ids[0]]}'!")
    m = trainer.compute_metrics((pred.predictions, pred.label_ids))
    m = {f"{prefix}_{k}": float(v) for k, v in m.items()}
    m[f"{prefix}_collapsed"] = bool(collapsed)
    m[f"{prefix}_pred_dist"] = pd_
    m[f"{prefix}_true_dist"] = td_
    return m


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()

    out_json = human_supervised_DIR / f"{args.task}_human_supervised_fold{args.fold}.json"
    if out_json.exists():
        print(f"[skip] {out_json.name} already exists.")
        return

    run_seed = SEED + args.fold        # same per-fold offset scheme as student
    set_seed(run_seed)

    task, cfg = args.task, TASKS[args.task]
    hp = HYPERPARAMS[task]
    num_labels = len(cfg["labels"])
    id2label = {i: l for i, l in ID2LABEL[task].items()}

    assert_disjoint(task)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL, num_labels=num_labels, id2label=id2label,
        label2id={l: i for l, i in LABEL2ID[task].items()},
    )

    # ---- Build sets: 6-fold on gold_train; fixed gold_eval + full bench --
    gold_train, _ = load_gold_train(task)
    gold_train = assign_folds(gold_train, N_FOLDS, SEED, "final_label")
    train_df   = gold_train.filter(pl.col("fold") != args.fold)
    heldout_df = gold_train.filter(pl.col("fold") == args.fold)
    gold_df    = load_gold_eval(task)
    bench_df   = load_benchmark(task)

    train_ds, y_train     = to_hf(train_df, task, tokenizer, "train")
    heldout_ds, y_heldout = to_hf(heldout_df, task, tokenizer, "heldout")
    gold_ds, y_gold       = to_hf(gold_df, task, tokenizer, "gold")

    print(f"[{task}/human_supervised/fold{args.fold}] seed={run_seed} "
          f"train={train_df.height} heldout={heldout_df.height} "
          f"gold={gold_df.height} "
          f"bench={bench_df.height if bench_df is not None else 0}")
    print(f"  train class dist: "
          + ", ".join(f"{id2label[i]}={c}"
                      for i, c in enumerate(class_dist(y_train, num_labels))))

    # ---- Class weights (identical to student) ----------------------------
    class_weights = None
    if USE_CLASS_WEIGHTS:
        present = np.unique(y_train)
        if len(present) < num_labels:
            print(f"  [warn] only {len(present)}/{num_labels} classes present")
        cw = compute_class_weight("balanced", classes=present, y=y_train)
        full = np.ones(num_labels, dtype=np.float32)
        for cls, w in zip(present, cw):
            full[cls] = w
        class_weights = torch.tensor(full, dtype=torch.float32)
        print("  class weights: "
              + ", ".join(f"{id2label[i]}={full[i]:.2f}"
                          for i in range(num_labels)))

    # ---- Training args (identical to student) ----------------------------
    output_dir = f"/tmp/s6_human_supervised_{task}_f{args.fold}"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    collator = DataCollatorWithPadding(tokenizer)
    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp["epochs"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="linear",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=25,
        seed=run_seed,
        data_seed=run_seed,
        report_to="none",
        bf16=use_bf16,
        fp16=False,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights, label_smoothing=LABEL_SMOOTHING,
        model=model, args=targs, train_dataset=train_ds,
        data_collator=collator, compute_metrics=make_metrics_fn(num_labels),
    )

    t0 = time.time()
    train_out = trainer.train()
    train_min = (time.time() - t0) / 60
    final_loss = float(train_out.training_loss)
    print(f"  final train_loss={final_loss:.4f}")

    heldout_m = eval_with_diag(trainer, heldout_ds, y_heldout, num_labels,
                               "heldout", id2label)
    gold_m = eval_with_diag(trainer, gold_ds, y_gold, num_labels,
                            "gold", id2label)
    if bench_df is not None:
        bench_ds, y_bench = to_hf(bench_df, task, tokenizer, "benchmark")
        bench_m = eval_with_diag(trainer, bench_ds, y_bench, num_labels,
                                 "bench", id2label)
    else:
        bench_m = {}

    result = {
        "task": task, "source": "gold_human", "size": train_df.height,
        "fold": args.fold, "seed": run_seed, "n_folds_config": N_FOLDS,
        "condition": "B4_human_supervised",
        "n_train": train_df.height, "n_heldout": heldout_df.height,
        "n_gold_eval": gold_df.height,
        "n_benchmark": bench_df.height if bench_df is not None else 0,
        "student_model": STUDENT_MODEL, "hyperparams": hp,
        "warmup_ratio": WARMUP_RATIO, "max_grad_norm": MAX_GRAD_NORM,
        "class_weighting": USE_CLASS_WEIGHTS, "label_smoothing": LABEL_SMOOTHING,
        "final_train_loss": final_loss, "train_minutes": round(train_min, 2),
        "train_class_dist": class_dist(y_train, num_labels),
        **heldout_m, **gold_m, **bench_m,
    }

    out_json.write_text(json.dumps(result, indent=2))
    shutil.rmtree(output_dir, ignore_errors=True)

    gf = result.get("gold_macro_f1")
    bf = result.get("bench_macro_f1")
    warn = " [!! GOLD COLLAPSED]" if result.get("gold_collapsed") else ""
    print(f"[done] gold_f1={gf:.4f}"
          + (f" bench_f1={bf:.4f}" if bf is not None else "")
          + f" loss={final_loss:.3f} ({train_min:.1f} min){warn}"
          + f" -> {out_json.name}")


if __name__ == "__main__":
    main()
