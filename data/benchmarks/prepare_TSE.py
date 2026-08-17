"""Prepare the TSE held-out benchmark for the sentiment generalization eval.
TSE is fully held out (retrieval index uses SemEval, not TSE) -> use ALL rows.
Outputs data/stage5_benchmarks/sentiment_benchmark.parquet:
  id, text, label_str  (label_str in {"POSITIVE","NEGATIVE","NEUTRAL"})
"""
import polars as pl
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

OUT_DIR = Path("~/data/stage5_benchmarks").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "sentiment_benchmark.parquet"

LABEL_MAP = {"negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE"}

def main():
    ds = load_dataset("mteb/tweet_sentiment_extraction")
    # Combine ALL splits (train + test) since TSE is fully held out
    combined = concatenate_datasets([ds[s] for s in ds.keys()])
    df = pl.from_pandas(combined.to_pandas())
    print(f"TSE rows (all splits): {df.height:,}")
    print("columns:", df.columns)

    df = df.with_columns(
        pl.col("label_text").str.strip_chars().str.to_lowercase()
          .replace_strict(LABEL_MAP, default=None)
          .alias("label_str")
    ).filter(
        pl.col("text").is_not_null() & pl.col("label_str").is_not_null()
    )

    out = df.select([
        pl.col("id").cast(pl.Utf8),
        pl.col("text"),
        pl.col("label_str"),
    ])
    print("Label distribution:")
    print(out.group_by("label_str").len().sort("len", descending=True))
    out.write_parquet(OUT, compression="zstd")
    print(f"Wrote {out.height:,} rows -> {OUT}")

if __name__ == "__main__":
    main()
