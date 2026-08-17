"""Stage 3c (news): class-balance each task's scored news sentences into a
final pool."""
import os, argparse
from pathlib import Path
import polars as pl

ROOT = Path(os.path.expanduser("~/"))
POOLS = ROOT / "stage3_pools"

# per-class caps — tune to reach your scaling ceiling (128k/task total) [[3]].
# For a 3-class task, 47k/class ~= 141k total, matching the twitter hate cap [[1]].
CAPS = {"hate": 47_000, "sexism": 64_000, "lexical_bias": 64_000, "sentiment": 47_000}

def build(task):
    scored = pl.read_parquet(POOLS / f"news_{task}_scored.parquet")
    pred_col, conf_col = f"{task}_pred", f"{task}_conf"
    cap = CAPS[task]
    classes = sorted(scored[pred_col].unique().to_list())
    print(f"[{task}] classes={classes}, available:")
    print(scored.group_by(pred_col).len().sort(pred_col))

    per_class = min(cap, *[scored.filter(pl.col(pred_col) == c).height for c in classes])
    print(f"[{task}] balanced per-class target: {per_class:,}")

    def top(c):
        return scored.filter(pl.col(pred_col) == c).sort(conf_col, descending=True).head(per_class)

    balanced = pl.concat([top(c) for c in classes], how="vertical_relaxed") \
                 .sample(fraction=1.0, seed=42, shuffle=True)

    out = POOLS / f"news_{task}_pool_balanced.parquet"
    balanced.write_parquet(out, compression="zstd", compression_level=3)
    print(f"[{task}] final pool: {balanced.height:,} -> {out}")
    print(balanced.group_by(pred_col).len().sort(pred_col))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(CAPS), default="all")
    args = ap.parse_args()
    tasks = list(CAPS) if args.task == "all" else [args.task]
    for t in tasks:
        build(t)

if __name__ == "__main__":
    main()
