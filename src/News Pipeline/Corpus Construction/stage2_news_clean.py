"""Stage 2 (news): clean + dedup sentence-level news data.
Mirrors twitter stage1/stage2 conventions: text_clean canonical field,
length filter, dedup on text_clean, zstd-3 output."""
import os, re
from pathlib import Path
import polars as pl

ROOT = Path(os.path.expanduser("~/"))
IN  = ROOT / "stage1_files" / "news_sentences.parquet"
OUT = ROOT / "stage2_files" / "news_sentences_clean.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

MIN_LEN, MAX_LEN = 20, 400   # min_text_length mirrors your >=20 filter [[6]] and consensus/length filter [[3]]

WS = re.compile(r"\s+")

def main():
    df = pl.read_parquet(IN)
    print(f"In: {df.height:,} sentences")

    # ensure canonical text_clean field exists (splitter produced 'text_clean')
    df = df.with_columns(
        pl.col("text_clean").str.replace_all(r"\s+", " ").str.strip_chars().alias("text_clean")
    )
    df = df.filter(pl.col("text_clean").is_not_null())

    # length filter (chars), mirrors stage1 min/max_text_length [[6]]
    df = df.filter(
        (pl.col("text_clean").str.len_chars() >= MIN_LEN)
        & (pl.col("text_clean").str.len_chars() <= MAX_LEN)
    )
    print(f"After length filter: {df.height:,}")

    # dedup on text_clean, keep first — same as stage2_dedupe [[2]]
    df = df.unique(subset=["text_clean"], keep="first")
    print(f"After dedup: {df.height:,}")

    df.write_parquet(OUT, compression="zstd", compression_level=3)
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()
