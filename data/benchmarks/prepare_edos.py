
import polars as pl
from pathlib import Path

SRC = Path("~/Gold Datasets/edos_labelled_aggregated.csv"
           ).expanduser()
OUT_DIR = Path("~/data/stage5_benchmarks").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "sexism_benchmark.parquet"

LABEL_MAP = {"sexist": "SEXIST", "not sexist": "NOT SEXIST"}

def main():
    df = pl.read_csv(SRC, infer_schema_length=10000, ignore_errors=True)
    print(f"EDOS rows (full, all splits): {df.height:,}")
    print(df.group_by("split").len().sort("split"))   # informational only

    # Fully held out -> use ALL rows (train/dev/test combined)
    df = df.with_columns(
        pl.col("label_sexist").str.strip_chars().str.to_lowercase()
          .replace_strict(LABEL_MAP, default=None)
          .alias("label_str")
    ).filter(
        pl.col("text").is_not_null() & pl.col("label_str").is_not_null()
    )

    out = df.select([
        pl.col("rewire_id").alias("id"),
        pl.col("text"),
        pl.col("label_str"),
    ])
    print("Label distribution (all rows):")
    print(out.group_by("label_str").len().sort("len", descending=True))
    out.write_parquet(OUT, compression="zstd")
    print(f"Wrote {out.height:,} rows -> {OUT}")

if __name__ == "__main__":
    main()
