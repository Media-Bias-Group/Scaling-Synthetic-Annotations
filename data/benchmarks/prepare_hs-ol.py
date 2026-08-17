
import polars as pl
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("~/data/stage5_benchmarks").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "hate_benchmark.parquet"

# class: 0 = hate speech, 1 = offensive language, 2 = neither  [[9]]
CLASS_MAP = {0: "HATE SPEECH", 1: "OFFENSIVE", 2: "NEITHER"}

def main():
    ds = load_dataset("tdavidson/hate_speech_offensive")
    # Dataset ships a single 'train' split; use ALL of it (fully held out).
    df = pl.from_pandas(ds["train"].to_pandas())
    print(f"HS-OL rows (full): {df.height:,}")
    print(df.columns)

    df = df.with_columns(
        pl.col("class").replace_strict(CLASS_MAP, default=None)
          .alias("label_str")
    ).filter(
        pl.col("tweet").is_not_null() & pl.col("label_str").is_not_null()
    )

    out = df.select([
        pl.arange(0, df.height).alias("id"),   # HS-OL has no native id column
        pl.col("tweet").alias("text"),
        pl.col("label_str"),
    ])
    print("Label distribution:")
    print(out.group_by("label_str").len().sort("len", descending=True))
    out.write_parquet(OUT, compression="zstd")
    print(f"Wrote {out.height:,} rows -> {OUT}")

if __name__ == "__main__":
    main()
