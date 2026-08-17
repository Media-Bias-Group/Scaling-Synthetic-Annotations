"""Stage 4d — Assemble final labels per task.
Final label = agreed teacher label where Mistral==gemma, else Claude tiebreak."""
import argparse
import polars as pl
from stage4_config_news import TASKS, OUT_DIR


def finalize(task):
    m = pl.read_parquet(OUT_DIR / f"{task}_mistral.parquet")
    g = pl.read_parquet(OUT_DIR / f"{task}_gemma.parquet")
    tb = pl.read_parquet(OUT_DIR / f"{task}_tiebreak.parquet")

    # Base join of the two teachers
    df = (m.select(["id", "text_clean", "mistral_label"])
            .join(g.select(["id", "gemma_label"]), on="id", how="inner"))

    # Cast tiebreak id to match the teacher id dtype (Anthropic returns str custom_id)
    id_dtype = df.schema["id"]
    tb = tb.with_columns(pl.col("id").cast(id_dtype, strict=False))

    # Attach claude labels
    df = df.join(tb.select(["id", "claude_label"]), on="id", how="left")

    # Final label logic
    df = df.with_columns(
        pl.when(pl.col("mistral_label") == pl.col("gemma_label"))
          .then(pl.col("mistral_label"))
          .otherwise(pl.col("claude_label"))
          .alias("final_label")
    )

    # Provenance: how was each final label decided?
    df = df.with_columns(
        pl.when(pl.col("mistral_label") == pl.col("gemma_label"))
          .then(pl.lit("teacher_agree"))
          .otherwise(pl.lit("claude_tiebreak"))
          .alias("label_source")
    )

    # --- Sanity checks ---
    total = df.height
    agree = df.filter(pl.col("label_source") == "teacher_agree").height
    tie = df.filter(pl.col("label_source") == "claude_tiebreak").height
    null_final = df.filter(pl.col("final_label").is_null()).height
    # Disagreements that got no claude label (join miss / parse fail)
    missing_tie = df.filter(
        (pl.col("mistral_label") != pl.col("gemma_label"))
        & pl.col("claude_label").is_null()
    ).height

    print("=" * 60)
    print(f"TASK: {task}")
    print("=" * 60)
    print(f"Total instances:        {total:,}")
    print(f"Teacher agreement:      {agree:,} ({agree/total*100:.2f}%)")
    print(f"Claude tiebreak:        {tie:,} ({tie/total*100:.2f}%)")
    print(f"NULL final labels:      {null_final:,}")
    print(f"Disagreements w/o tiebreak label: {missing_tie:,}")
    print("\nFinal label distribution:")
    print(df.group_by("final_label").agg(pl.len().alias("count"))
            .sort("count", descending=True))

    out = df.select(["id", "text_clean", "mistral_label", "gemma_label",
                     "claude_label", "final_label", "label_source"])
    out_path = OUT_DIR / f"{task}_final.parquet"
    out.write_parquet(out_path)
    print(f"\nSaved: {out_path}")
    return null_final, missing_tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    args = ap.parse_args()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    issues = 0
    for t in tasks:
        nf, mt = finalize(t)
        issues += nf + mt
        print()

    if issues == 0:
        print("✓ All tasks finalized with zero null/missing labels.")
    else:
        print(f"⚠ {issues} null/missing labels total — investigate (likely "
              f"id dtype mismatch or parse failures in tiebreak).")


if __name__ == "__main__":
    main()
