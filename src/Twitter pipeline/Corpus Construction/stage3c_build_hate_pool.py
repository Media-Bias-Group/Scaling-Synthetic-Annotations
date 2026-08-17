"""Combine the original hate pool with newly-mined hateful instances,
then build a class-balanced final hate pool."""

import os
from pathlib import Path
import polars as pl

ROOT = Path(os.path.expanduser("~/data"))
POOLS = ROOT / "stage3_pools"
ORIG = POOLS / "hate_pool.parquet"
NEW  = POOLS / "hate_topup_hateful.parquet"
OUT  = POOLS / "hate_pool_balanced.parquet"

FOL = "author.public_metrics.followers_count"


def main():
    orig = pl.read_parquet(ORIG)
    new = pl.read_parquet(NEW)
    print(f"Original hate pool: {orig.height:,} rows")
    print(orig.group_by("hate_pred").len().sort("hate_pred"))
    print(f"\nNewly-mined hateful: {new.height:,} rows")

    orig_hateful = orig.filter(pl.col("hate_pred") == 0)
    common_cols = [c for c in orig_hateful.columns if c in new.columns]
    all_hateful = pl.concat(
        [orig_hateful.select(common_cols), new.select(common_cols)],
        how="vertical_relaxed",
    ).unique(subset=["id"], keep="first")
    print(f"\nCombined hateful (deduped on id): {all_hateful.height:,} rows")

    offensive = orig.filter(pl.col("hate_pred") == 1)
    normal    = orig.filter(pl.col("hate_pred") == 2)
    print(f"Offensive available: {offensive.height:,}")
    print(f"Normal available:    {normal.height:,}")

    per_class = min(all_hateful.height, offensive.height, normal.height, 47_000)
    print(f"\nBalanced per-class target: {per_class:,}")

    def top(df, col="hate_conf", n=per_class):
        return df.sort(col, descending=True).head(n)

    balanced = pl.concat(
        [top(all_hateful), top(offensive), top(normal)],
        how="vertical_relaxed",
    ).sample(fraction=1.0, seed=42, shuffle=True)

    balanced.write_parquet(OUT, compression="zstd", compression_level=3)
    print(f"\nFinal balanced hate pool: {balanced.height:,} rows")
    print(balanced.group_by("hate_pred").len().sort("hate_pred"))
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
