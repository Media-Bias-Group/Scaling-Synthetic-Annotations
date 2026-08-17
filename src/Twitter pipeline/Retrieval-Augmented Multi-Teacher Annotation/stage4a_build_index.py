"""Build the RAG retrieval index for any task (seeded 80/20 train/eval split).

Supports two gold formats:
  - CSV with a boolean/raw label column -> mapped via cfg["label_map"]
  - Parquet that already contains a "label_str" column (e.g. pre-processed BABE)
"""
import argparse
import numpy as np, polars as pl
from sentence_transformers import SentenceTransformer
from stage4_config import TASKS, EMBED_MODEL, INDEX_DIR, SPLIT_SEED, TRAIN_FRAC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sexism", choices=list(TASKS))
    args = ap.parse_args()

    TASK = args.task
    cfg = TASKS[TASK]

    gold_file = str(cfg["gold_file"])
    print(f"Task: {TASK}")
    print(f"Gold file: {gold_file}")

    # ---- Load gold (CSV or parquet) --------------------------------------
    if gold_file.endswith(".parquet"):
        df = pl.read_parquet(gold_file)
    else:
        # BABE uses ';' separator; most others use ','. Detect simply.
        sep = ";" if "babe" in gold_file.lower() else ","
        df = pl.read_csv(
            gold_file,
            separator=sep,
            quote_char='"',
            infer_schema_length=10000,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
    print(f"Gold rows: {df.height:,}")

    # ---- Ensure a label_str column exists --------------------------------
    if "label_str" not in df.columns:
        label_col = cfg["gold_label_col"]
        label_map = cfg["label_map"]
        # Cast to boolean when the mapping is keyed on True/False
        if set(label_map.keys()) == {True, False}:
            df = df.with_columns(
                pl.col(label_col).cast(pl.Boolean, strict=False)
            )
            df = df.with_columns(
                pl.col(label_col)
                  .map_elements(lambda b: label_map[bool(b)], return_dtype=pl.Utf8)
                  .alias("label_str")
            )
        else:
            # Generic mapping (e.g. string -> label string)
            df = df.with_columns(
                pl.col(label_col)
                  .map_elements(lambda v: label_map.get(v), return_dtype=pl.Utf8)
                  .alias("label_str")
            )

    # Drop rows without a usable text or label
    text_col = cfg["gold_text_col"]
    df = df.filter(
        pl.col(text_col).is_not_null() & pl.col("label_str").is_not_null()
    )
    print(f"Usable rows after cleaning: {df.height:,}")
    print("Label distribution:")
    print(df.group_by("label_str").agg(pl.len().alias("count"))
            .sort("count", descending=True))

    # ---- Seeded 80/20 split (retrieval draws ONLY from train) ------------
    df = df.sample(fraction=1.0, seed=SPLIT_SEED, shuffle=True)
    n_train = int(df.height * TRAIN_FRAC)
    train = df.head(n_train)
    eval_ = df.tail(df.height - n_train)
    train.write_parquet(INDEX_DIR / f"{TASK}_gold_train.parquet")
    eval_.write_parquet(INDEX_DIR / f"{TASK}_gold_eval.parquet")
    print(f"train: {train.height:,}  eval: {eval_.height:,}")

    # ---- Embed the train texts for retrieval -----------------------------
    texts = train[text_col].to_list()
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)
    np.save(INDEX_DIR / f"{TASK}_train_emb.npy", emb)
    print(f"Saved embeddings {emb.shape} -> {INDEX_DIR}")
    print(f"Index files written: {TASK}_gold_train.parquet, "
          f"{TASK}_gold_eval.parquet, {TASK}_train_emb.npy")


if __name__ == "__main__":
    main()
