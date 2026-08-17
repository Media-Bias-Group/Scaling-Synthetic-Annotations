"""Stage 3a: Draw a 5M proportional random subsample from the 12 monthly
deduplicated Parquet files. Runs locally. Fixed seed for reproducibility."""

import os
import glob
import time
from pathlib import Path

import polars as pl

LOCAL_ROOT = Path(os.path.expanduser("~/data"))
MONTHLY_DIR = LOCAL_ROOT / "stage2_monthly_merged"
OUTPUT_DIR = LOCAL_ROOT / "stage3_subsample"
OUTPUT_FILE = OUTPUT_DIR / "subsample_5M.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOTAL = 5_000_000
SEED = 42

FOLLOWERS_COL = "author.public_metrics.followers_count"


def main():
    start = time.time()
    files = sorted(glob.glob(str(MONTHLY_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"No monthly files found in {MONTHLY_DIR}")

    print(f"Found {len(files)} monthly files")

    # 1. Get per-file row counts (cheap, metadata-only scan)
    counts = {}
    for f in files:
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        counts[f] = n
        print(f"  {Path(f).stem}: {n:,} rows")
    grand_total = sum(counts.values())
    print(f"Total rows across months: {grand_total:,}")

    # 2. Proportional allocation per month
    frac = TARGET_TOTAL / grand_total
    print(f"Sampling fraction: {frac:.4%}")

    # 3. Sample each month proportionally, coercing schema for safe concat
    parts = []
    allocated = 0
    for i, f in enumerate(files):
        n = counts[f]
        # last file gets the remainder to hit exactly TARGET_TOTAL
        if i == len(files) - 1:
            take = TARGET_TOTAL - allocated
        else:
            take = int(round(n * frac))
        allocated += take

        df = (
            pl.scan_parquet(f)
            .with_columns(
                pl.col(FOLLOWERS_COL).cast(pl.Int64, strict=False).alias(FOLLOWERS_COL)
            )
            .collect()
        )
        # sample without replacement, capped at available rows
        take = min(take, df.height)
        sampled = df.sample(n=take, seed=SEED, with_replacement=False)
        parts.append(sampled)
        print(f"  {Path(f).stem}: sampled {take:,}")

    subsample = pl.concat(parts, how="vertical_relaxed")
    # final shuffle so months are interleaved
    subsample = subsample.sample(fraction=1.0, seed=SEED, shuffle=True)

    subsample.write_parquet(OUTPUT_FILE, compression="zstd", compression_level=3)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print("=" * 60)
    print(f"Subsample rows: {subsample.height:,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Elapsed: {(time.time()-start)/60:.1f} min")
    print("=" * 60)
    print("\nColumns:", subsample.columns)
    print("\nSample rows:")
    print(subsample.select(["id", "created_at", "text_clean"]).head(3))


if __name__ == "__main__":
    main()
