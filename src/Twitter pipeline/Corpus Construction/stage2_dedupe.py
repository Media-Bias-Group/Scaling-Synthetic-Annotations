"""Stage 2: Global deduplication via day-level chunking with Polars.
Two passes: (1) dedup within each day, (2) dedup across days in a final merge.
"""

import os
import re
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

LOCAL_ROOT = Path(os.path.expanduser("~/data"))
STAGE1_DIR = LOCAL_ROOT / "stage1_cleaned"
DAILY_DIR = LOCAL_ROOT / "stage2_daily"
OUTPUT_DIR = LOCAL_ROOT / "stage2_deduplicated"
OUTPUT_FILE = OUTPUT_DIR / "tweets_deduplicated.parquet"

DAILY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_\d{2}_tweets\.parquet$")


def group_files_by_day():
    """Map each date -> list of hourly parquet files for that day."""
    days = defaultdict(list)
    for p in STAGE1_DIR.glob("*.parquet"):
        m = DATE_RE.match(p.name)
        if m:
            days[m.group(1)].append(p)
    return dict(sorted(days.items()))


def main():
    start = time.time()
    days = group_files_by_day()
    print(f"Found {len(days)} days of data")

    # ---- Pass 1: dedup within each day ----
    print("\nPASS 1: per-day deduplication")
    total_in = 0
    for i, (day, files) in enumerate(days.items(), 1):
        out = DAILY_DIR / f"{day}.parquet"
        if out.exists() and out.stat().st_size > 0:
            continue  # resume-safe

        # Read all hourly files for this day (lazy), concat, dedup on text_clean
        lfs = [pl.scan_parquet(str(f)) for f in files]
        lf = pl.concat(lfs, how="vertical_relaxed")
        lf = lf.filter(pl.col("text_clean").is_not_null())
        # Keep first occurrence per unique cleaned text within the day
        df = lf.collect(streaming=True)
        n_in = df.height
        df = df.unique(subset=["text_clean"], keep="first")
        df.write_parquet(out, compression="zstd", compression_level=3)
        total_in += n_in

        if i % 20 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] {day}: {n_in:,} -> {df.height:,}")

    # ---- Pass 2: merge all days and dedup across day boundaries ----
    print("\nPASS 2: cross-day merge + final dedup")
    # Stream all daily files, dedup globally. Since each day is already deduped,
    # the working set is far smaller than the raw 257M.
    daily_files = sorted(DAILY_DIR.glob("*.parquet"))
    print(f"  Merging {len(daily_files)} daily files...")

    lf = pl.scan_parquet(str(DAILY_DIR / "*.parquet"))
    lf = lf.unique(subset=["text_clean"], keep="first")
    # sink_parquet streams the result to disk without materializing all in RAM
    lf.sink_parquet(str(OUTPUT_FILE), compression="zstd")

    # Count output
    total_out = pl.scan_parquet(str(OUTPUT_FILE)).select(pl.len()).collect().item()

    elapsed = time.time() - start
    removed = total_in - total_out
    print("=" * 60)
    print(f"Input rows (sum of days):   {total_in:,}")
    print(f"Unique rows out:            {total_out:,}")
    if total_in:
        print(f"Duplicates removed:         {removed:,} ({100*removed/total_in:.2f}%)")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
