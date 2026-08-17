"""Draw a 40M top-up subsample from the 12 monthly deduplicated files,
EXCLUDING ids already present in the original 5M subsample. This gives
fresh tweets for mining additional hateful instances."""

import os
import glob
import time
from pathlib import Path

import polars as pl

ROOT = Path(os.path.expanduser("~/data"))
MONTHLY = ROOT / "stage2_monthly_merged"
PREV = ROOT / "stage3_subsample" / "subsample_5M.parquet"
OUT = ROOT / "stage3_subsample" / "subsample_topup_40M.parquet"

TARGET = 40_000_000
SEED = 43
FOL = "author.public_metrics.followers_count"


def main():
    start = time.time()

    # Load already-seen ids as a small DataFrame for anti-join (memory-efficient)
    print("Loading previously-tagged ids for exclusion...")
    seen_df = pl.read_parquet(PREV, columns=["id"])
    print(f"  {seen_df.height:,} ids to exclude")

    files = sorted(glob.glob(str(MONTHLY / "*.parquet")))
    print(f"Found {len(files)} monthly files")

    # Per-month row counts (metadata only)
    counts = {}
    for f in files:
        counts[f] = pl.scan_parquet(f).select(pl.len()).collect().item()
    grand_total = sum(counts.values())
    print(f"Total monthly rows: {grand_total:,}")

    remaining = grand_total - seen_df.height
    frac = min(1.0, TARGET / remaining)
    print(f"Sampling fraction (of remaining): {frac:.4%}")

    parts = []
    allocated = 0
    for i, f in enumerate(files):
        n = counts[f]
        # anti-join to exclude seen ids; light on memory since seen_df is tiny (~5M ids)
        df = (
            pl.scan_parquet(f)
            .with_columns(pl.col(FOL).cast(pl.Int64, strict=False).alias(FOL))
            .join(seen_df.lazy(), on="id", how="anti")
            .collect()
        )

        if i == len(files) - 1:
            take = TARGET - allocated
        else:
            take = int(round(n * frac))
        take = min(take, df.height)
        allocated += take

        sampled = df.sample(n=take, seed=SEED, with_replacement=False)
        parts.append(sampled)
        print(f"  {Path(f).stem}: available {df.height:,}, sampled {take:,}")

    top = pl.concat(parts, how="vertical_relaxed")
    top = top.sample(fraction=1.0, seed=SEED, shuffle=True)
    top.write_parquet(OUT, compression="zstd", compression_level=3)

    size_gb = OUT.stat().st_size / (1024**3)
    print("=" * 60)
    print(f"Top-up subsample: {top.height:,} rows")
    print(f"Output: {OUT} ({size_gb:.2f} GB)")
    print(f"Elapsed: {(time.time()-start)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
