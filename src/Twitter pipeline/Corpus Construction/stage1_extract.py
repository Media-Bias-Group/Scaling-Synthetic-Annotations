"""Stage 1: Streaming extraction and initial filtering of Twitter data."""

from __future__ import annotations

import argparse
import io
import lzma
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import yaml
from tqdm import tqdm


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@(\w{1,15})")
HASHTAG_RE = re.compile(r"#\w+")
RT_PREFIX_RE = re.compile(r"^RT[\s:]+", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text):
    if text is None or not isinstance(text, str):
        return None
    text = RT_PREFIX_RE.sub("", text)
    text = MENTION_RE.sub("@MENTION", text)
    text = URL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text if text else None


def url_ratio(original_text):
    if not original_text:
        return 0.0
    url_chars = sum(len(m.group(0)) for m in URL_RE.finditer(original_text))
    return url_chars / max(len(original_text), 1)


def mention_ratio(original_text):
    if not original_text:
        return 0.0
    m_chars = sum(len(m.group(0)) for m in MENTION_RE.finditer(original_text))
    return m_chars / max(len(original_text), 1)


def hashtag_count(original_text):
    if not original_text:
        return 0
    return len(HASHTAG_RE.findall(original_text))


KEEP_COLUMNS = [
    "id", "created_at", "lang", "text", "reference_type",
    "author_id", "author.public_metrics.followers_count",
]


def read_xz_tsv(input_path, max_rows=None):
    """
    Decompress .xz and decode bytes with error tolerance, then hand clean
    UTF-8 text to Polars. This avoids 'invalid utf-8 sequence' errors from
    malformed bytes common in scraped Twitter data.
    """
    with lzma.open(input_path, "rb") as fh:
        raw_bytes = fh.read()

    # Decode with error replacement: invalid bytes become the replacement char
    text = raw_bytes.decode("utf-8", errors="replace")

    # Optionally truncate to first max_rows+1 lines (for smoke test) before parsing
    if max_rows is not None:
        lines = text.split("\n")
        # keep header + max_rows data lines
        text = "\n".join(lines[: max_rows + 1])

    buf = io.StringIO(text)

    df = pl.read_csv(
        buf,
        separator="\t",
        has_header=True,
        ignore_errors=True,
        null_values=["", "null", "NULL", "\\N"],
        infer_schema_length=1000,
        truncate_ragged_lines=True,
        quote_char=None,          # tweets contain unbalanced quotes; disable quote parsing
        encoding="utf8",
    )
    return df


def process_file(input_path, output_path, config, max_rows=None):
    stats = {
        "file": input_path.name, "rows_in": 0, "rows_after_lang": 0,
        "rows_after_refs": 0, "rows_after_text": 0, "rows_after_spam": 0,
        "rows_after_length": 0, "rows_out": 0, "elapsed_s": 0.0,
        "status": "ok", "error": None,
    }
    start = time.time()

    try:
        df = read_xz_tsv(input_path, max_rows=max_rows)

        cols_to_keep = [c for c in KEEP_COLUMNS if c in df.columns]
        df = df.select(cols_to_keep)
        stats["rows_in"] = df.height
        if df.height == 0:
            stats["status"] = "empty"; return stats

        if "lang" in df.columns:
            df = df.filter(pl.col("lang").is_in(config["allowed_languages"]))
        stats["rows_after_lang"] = df.height
        if df.height == 0:
            stats["status"] = "empty_after_lang"; return stats

        if config["drop_retweets"] and "reference_type" in df.columns:
            df = df.filter(
                ~pl.col("reference_type").is_in(config["drop_reference_types"])
                | pl.col("reference_type").is_null()
            )
        stats["rows_after_refs"] = df.height

        df = df.filter(pl.col("text").is_not_null() & (pl.col("text").str.len_chars() > 0))
        stats["rows_after_text"] = df.height
        if df.height == 0:
            stats["status"] = "empty_after_text"; return stats

        df = df.with_columns([
            pl.col("text").map_elements(url_ratio, return_dtype=pl.Float64).alias("_url_ratio"),
            pl.col("text").map_elements(mention_ratio, return_dtype=pl.Float64).alias("_mention_ratio"),
            pl.col("text").map_elements(hashtag_count, return_dtype=pl.Int32).alias("_hashtag_count"),
        ])

        df = df.filter(
            (pl.col("_url_ratio") <= config["max_url_ratio"])
            & (pl.col("_mention_ratio") <= config["max_mention_ratio"])
            & (pl.col("_hashtag_count") <= config["max_hashtag_count"])
        )
        stats["rows_after_spam"] = df.height
        if df.height == 0:
            stats["status"] = "empty_after_spam"; return stats

        df = df.with_columns(
            pl.col("text").map_elements(normalize_text, return_dtype=pl.Utf8).alias("text_clean")
        )

        df = df.filter(
            pl.col("text_clean").is_not_null()
            & (pl.col("text_clean").str.len_chars() >= config["min_text_length"])
            & (pl.col("text_clean").str.len_chars() <= config["max_text_length"])
        )
        stats["rows_after_length"] = df.height

        df = df.unique(subset=["text_clean"], keep="first")
        stats["rows_out"] = df.height
        if df.height == 0:
            stats["status"] = "empty_after_dedup"; return stats

        drop_cols = [c for c in ["_url_ratio", "_mention_ratio", "_hashtag_count"] if c in df.columns]
        df = df.drop(drop_cols)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, compression="zstd", compression_level=3)

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = f"{type(e).__name__}: {e}"

    stats["elapsed_s"] = round(time.time() - start, 2)
    return stats


def _worker(args):
    input_path, output_path, config, max_rows = args
    return process_file(input_path, output_path, config, max_rows)


def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stage1_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return log_file


def discover_input_files(input_dir):
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}_tweets\.tsv\.xz$")
    return sorted(p for p in input_dir.iterdir() if pattern.match(p.name))


def output_path_for(input_file, output_dir):
    stem = input_file.name.replace(".tsv.xz", "")
    return output_dir / f"{stem}.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mount = Path(os.path.expanduser(config["nextcloud_mount"]))
    input_dir = mount / config["input_subpath"]
    local_root = Path(os.path.expanduser(config["local_output_root"]))
    output_dir = local_root / config["output_subdir"]
    log_dir = local_root / config["log_subdir"]

    log_file = setup_logging(log_dir)
    logging.info(f"Input dir: {input_dir}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Log file: {log_file}")

    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    smoke = args.smoke_test or config.get("smoke_test", {}).get("enabled", False)
    max_rows = None
    if smoke:
        max_rows = config.get("smoke_test", {}).get("max_rows_per_file", 5000)
        logging.info(f"SMOKE TEST MODE: max_rows_per_file={max_rows}")

    all_files = discover_input_files(input_dir)
    logging.info(f"Discovered {len(all_files)} input files")

    if smoke:
        n = config.get("smoke_test", {}).get("num_files", 3)
        all_files = all_files[:n]
        logging.info(f"Smoke test: limiting to first {n} files")
    elif args.limit:
        all_files = all_files[:args.limit]

    skip_existing = not args.no_skip_existing
    if skip_existing and not smoke:
        files_to_process = [
            f for f in all_files
            if not (output_path_for(f, output_dir).exists()
                    and output_path_for(f, output_dir).stat().st_size > 0)
        ]
        skipped = len(all_files) - len(files_to_process)
        if skipped:
            logging.info(f"Skipping {skipped} already-processed files")
    else:
        files_to_process = all_files

    if not files_to_process:
        logging.info("Nothing to do.")
        return

    logging.info(f"Processing {len(files_to_process)} files with {config['num_workers']} workers")
    work_items = [(f, output_path_for(f, output_dir), config, max_rows) for f in files_to_process]

    all_stats, total_in, total_out, errors = [], 0, 0, 0

    with ProcessPoolExecutor(max_workers=config["num_workers"]) as executor:
        futures = {executor.submit(_worker, item): item[0] for item in work_items}
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                stats = future.result()
                all_stats.append(stats)
                total_in += stats["rows_in"]
                total_out += stats["rows_out"]
                if stats["status"] == "error":
                    errors += 1
                    logging.error(f"FAILED {stats['file']}: {stats['error']}")
                else:
                    logging.info(
                        f"{stats['file']}: {stats['rows_in']:>7} -> {stats['rows_out']:>7} "
                        f"({stats['elapsed_s']}s) [{stats['status']}]"
                    )
                pbar.update(1)

    stats_df = pl.DataFrame(all_stats)
    stats_path = output_dir / f"_stats_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
    stats_df.write_parquet(stats_path)

    logging.info("=" * 60)
    logging.info(f"Files processed: {len(all_stats)}")
    logging.info(f"Files with errors: {errors}")
    logging.info(f"Total rows in: {total_in:,}")
    logging.info(f"Total rows out: {total_out:,}")
    if total_in > 0:
        logging.info(f"Retention rate: {100 * total_out / total_in:.2f}%")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
