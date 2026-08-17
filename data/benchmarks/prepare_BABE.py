import polars as pl
from pathlib import Path

df = pl.read_csv("/Gold Datasets/Babe/final_labels_SG2.csv",
                 separator="\t", ignore_errors=True)
df = df.with_columns(
    pl.when(pl.col("label_bias") == "Biased").then(pl.lit("BIASED"))
      .otherwise(pl.lit("NOT BIASED")).alias("label_str")
).select(["text", "label_str"])


df.write_parquet("/Gold Datasets/Babe/lexical_bias_gold.parquet")
print(df.height, "rows"); print(df["label_str"].value_counts())
