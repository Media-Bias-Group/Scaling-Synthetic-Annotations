"""Stage 5a — Build nested subsamples + 6-fold stratified CV assignments.

For each task and each size N in SIZES, draw N instances from the task's
pool (<task>_final.parquet for twitter, <task>_<source>_final.parquet
otherwise). Sizes are NESTED (1k subset of 4k subset of ... of 128k) so
scaling curves are not confounded by which instances were drawn. Within each
size, assign N_FOLDS stratified folds on final_label.

MIXED condition (source="mixed"): a 50:50 blend of the twitter and news pools
at MATCHED TOTAL size (Option A). At size N, draw N/2 from each source (nested
within each source), tag rows with a per-row `source` column, and assign folds
stratified on source x final_label so every fold is ~50:50 source AND preserves
each source's class proportions.

Saves lightweight index files per (task, source, size):
  stage5_splits/{task}_{source}_{size}_folds.parquet
    single-source: (id, text_clean, final_label, fold)
    mixed:         (id, text_clean, final_label, source, fold)

Fixed seed (SEED=42) throughout, matching stage3a/stage3c conventions.

Usage:
  python stage5a_build_splits.py                          # all sources in SOURCES, all tasks
  python stage5a_build_splits.py --source news            # only news, all tasks
  python stage5a_build_splits.py --source news --task lexical_bias   # one task
  python stage5a_build_splits.py --source mixed --task sentiment     # mixed 50:50
"""
import argparse
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from stage5_config import (TASKS, SIZES, SOURCES, N_FOLDS, SEED,
                           NESTED_SAMPLING, SPLIT_DIR)


def load_pool(task, source):
    """Load the Stage 4d final pool for (task, source); keep id, text, label.

    twitter -> <task>_final.parquet
    other   -> <task>_<source>_final.parquet   (e.g. lexical_bias_news_final.parquet)
    The pool directory is inferred from the task's configured final_file.
    """
    cfg = TASKS[task]
    suffix = "" if source == "twitter" else f"_{source}"
    pool_path = cfg["final_file"].parent / f"{task}{suffix}_final.parquet"
    if not pool_path.exists():
        raise SystemExit(f"[{task}/{source}] pool not found: {pool_path}")
    df = pl.read_parquet(pool_path)
    # Expects columns from stage4d finalize: id, text_clean, final_label, label_source
    needed = ["id", "text_clean", "final_label", "label_source"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"[{task}/{source}] pool missing columns {missing}. "
                         f"Have: {df.columns}")
    df = df.filter(
        pl.col("text_clean").is_not_null()
        & pl.col("final_label").is_not_null()
    )
    return df


def nested_sample(df, sizes, seed):
    """Return dict size -> DataFrame, with each smaller size a subset of the
    next larger one. We shuffle once, then take prefixes."""
    shuffled = df.sample(fraction=1.0, seed=seed, shuffle=True)
    max_size = max(sizes)
    if shuffled.height < max_size:
        print(f"  WARNING: pool has {shuffled.height:,} < {max_size:,} "
              f"requested; capping sizes to available rows.")
    out = {}
    for n in sorted(sizes):
        take = min(n, shuffled.height)
        out[n] = shuffled.head(take)
    return out


def independent_sample(df, sizes, seed):
    """Fallback: independent draw per size (not nested)."""
    out = {}
    for n in sizes:
        take = min(n, df.height)
        out[n] = df.sample(n=take, seed=seed, with_replacement=False)
    return out


def assign_folds(df, n_folds, seed, strat_col="final_label"):
    """Return the df with an added integer 'fold' column, stratified on
    strat_col. Falls back to non-stratified split if a class is too rare."""
    keys = df[strat_col].to_numpy()
    idx = np.arange(len(keys))
    fold_col = np.full(len(keys), -1, dtype=np.int64)

    # StratifiedKFold preserves per-key proportions across folds.
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed)
        for k, (_, test_idx) in enumerate(skf.split(idx, keys)):
            fold_col[test_idx] = k
    except ValueError as e:
        # e.g. a class has fewer members than n_folds
        print(f"    stratified split failed ({e}); using plain KFold")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (_, test_idx) in enumerate(kf.split(idx)):
            fold_col[test_idx] = k

    return df.with_columns(pl.Series("fold", fold_col))


def build(task, source):
    """Single-source (twitter / news / ...) split builder."""
    print(f"\n=== {task} / {source} ===")
    pool = load_pool(task, source)
    print(f"  pool rows: {pool.height:,}")
    print(pool.group_by("final_label").len().sort("final_label"))

    samples = (nested_sample(pool, SIZES, SEED)
               if NESTED_SAMPLING
               else independent_sample(pool, SIZES, SEED))

    for size, sdf in samples.items():
        sdf = assign_folds(sdf, N_FOLDS, SEED, strat_col="final_label")
        out = SPLIT_DIR / f"{task}_{source}_{size}_folds.parquet"
        (sdf.select(["id", "text_clean", "final_label", "fold"])
            .write_parquet(out, compression="zstd"))
        dist = (sdf.group_by(["fold"]).len().sort("fold")["len"].to_list())
        print(f"  size={size:>6}: {sdf.height:>6} rows -> "
              f"folds {dist}  ({out.name})")


def build_mixed(task):
    """MIXED (50:50 twitter+news) at MATCHED TOTAL size (Option A).

    At size N: N/2 from twitter + N/2 from news, nested within each source,
    with a per-row `source` column, and folds stratified on source x label.
    """
    print(f"\n=== {task} / mixed (50:50 twitter+news) ===")
    twitter = (load_pool(task, "twitter")
               .with_columns(pl.lit("twitter").alias("source")))
    news = (load_pool(task, "news")
            .with_columns(pl.lit("news").alias("source")))
    print(f"  twitter pool: {twitter.height:,}   news pool: {news.height:,}")

    # shuffle each source ONCE so the per-source prefixes are nested
    tw = twitter.sample(fraction=1.0, seed=SEED, shuffle=True)
    nw = news.sample(fraction=1.0, seed=SEED, shuffle=True)

    for size in sorted(SIZES):
        half = size // 2
        take = min(half, tw.height, nw.height)
        if take < half:
            print(f"  [warn] size={size}: wanted {half}/source, capping to "
                  f"{take} (tw={tw.height}, nw={nw.height})")
        # 50:50 blend, nested prefixes within each source
        part = pl.concat([tw.head(take), nw.head(take)])

        # combined stratification key: source x label
        part = part.with_columns(
            (pl.col("source") + "|" + pl.col("final_label")).alias("_strat"))
        part = assign_folds(part, N_FOLDS, SEED, strat_col="_strat")

        out = SPLIT_DIR / f"{task}_mixed_{size}_folds.parquet"
        (part.select(["id", "text_clean", "final_label", "source", "fold"])
             .write_parquet(out, compression="zstd"))

        # report balance
        src_ct = {r[0]: r[1] for r in
                  part.group_by("source").len().sort("source").iter_rows()}
        fold_dist = (part.group_by("fold").len().sort("fold")["len"].to_list())
        print(f"  size={size:>6}: {part.height:>6} rows  sources={src_ct}  "
              f"folds {fold_dist}  ({out.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="single source (twitter/news/mixed); default = all in SOURCES")
    ap.add_argument("--task", default=None,
                    help="single task; default = all in TASKS")
    args = ap.parse_args()

    sources = [args.source] if args.source else list(SOURCES)
    tasks = [args.task] if args.task else list(TASKS)

    for source in sources:
        for task in tasks:
            if source == "mixed":
                build_mixed(task)
            else:
                build(task, source)

    print("\nDone. Split index files written to", SPLIT_DIR)


if __name__ == "__main__":
    main()
