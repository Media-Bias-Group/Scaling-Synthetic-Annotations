

import os
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

LOCAL_ROOT = Path(os.path.expanduser("~/data"))
DAILY_DIR = LOCAL_ROOT / "stage2_daily"
MONTHLY_DIR = LOCAL_ROOT / "stage2_monthly_merged"
OUTPUT_DIR = LOCAL_ROOT / "stage2_deduplicated"
OUTPUT_FILE = OUTPUT_DIR / "tweets_deduplicated.parquet"

MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLLOWERS_COL = "author.public_metrics.followers_count"


def coerce(lf):
    """Force the problematic column to Int64 so schemas align."""
    return lf.with_columns(
        pl.col(FOLLOWERS_COL).cast(pl.Int64, strict=False).alias(FOLLOWERS_COL)
    )


def main():
    start = time.time()

    print("STEP A: merge daily files into monthly deduped files")
    by_month = defaultdict(list)
    for f in sorted(DAILY_DIR.glob("*.parquet")):
        # filename like 2018-01-15.parquet -> month key 2018-01
        month = f.stem[:7]
        by_month[month].append(f)

    for month, files in sorted(by_month.items()):
        out = MONTHLY_DIR / f"{month}.parquet"
        if out.exists() and out.stat().st_size > 0:
            print(f"  {month}: already merged, skipping")
            continue

        t0 = time.time()
        lfs = [coerce(pl.scan_parquet(str(f))) for f in files]
        merged = pl.concat(lfs, how="vertical_relaxed")
        merged = merged.filter(pl.col("text_clean").is_not_null())
        merged = merged.unique(subset=["text_clean"], keep="first")
        # Materialize this month (small enough), then write
        df = merged.collect(engine="streaming")
        df.write_parquet(out, compression="zstd", compression_level=3)
        print(f"  {month}: {len(files)} days -> {df.height:,} rows ({(time.time()-t0)/60:.1f} min)")
        del df, merged, lfs

    print("\nSTEP B: merge monthly files into final deduped output")
    monthly_files = sorted(MONTHLY_DIR.glob("*.parquet"))
    lfs = [coerce(pl.scan_parquet(str(f))) for f in monthly_files]
    merged = pl.concat(lfs, how="vertical_relaxed")
    merged = merged.unique(subset=["text_clean"], keep="first")
    merged.sink_parquet(str(OUTPUT_FILE), compression="zstd")

    total_out = pl.scan_parquet(str(OUTPUT_FILE)).select(pl.len()).collect().item()
    total_in = 0
    for f in monthly_files:
        total_in += pl.scan_parquet(str(f)).select(pl.len()).collect().item()

    elapsed = time.time() - start
    removed = total_in - total_out
    print("=" * 60)
    print(f"Monthly rows in:          {total_in:,}")
    print(f"Unique rows out (global): {total_out:,}")
    if total_in:
        print(f"Cross-month duplicates:   {removed:,} ({100*removed/total_in:.2f}%)")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
