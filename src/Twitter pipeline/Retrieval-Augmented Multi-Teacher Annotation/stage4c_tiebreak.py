"""Stage 4c — Blind tiebreak of teacher disagreements via Claude Sonnet 5 (Batch API).

For each task:
  1. Joins the two teacher parquets on id, find disagreements.
  2. Builds a BLIND prompt per disagreement: task definition + k=8 retrieved
     examples + target text. Teacher labels are NOT shown (no anchoring).
  3. Submits all disagreements as one Anthropic Message Batch.
  4. Polls until complete, download results, parse the final label.
  5. Writes <task>_tiebreak.parquet (id, claude_label, claude_raw).

Run in two phases:
  python stage4c_tiebreak.py --task hate --submit    # creates + submits batch
  python stage4c_tiebreak.py --task hate --collect   # polls + downloads + parses
Or --run to do both (submit, then block-poll until done).
"""
import argparse, json, time, os
import numpy as np, polars as pl
from pathlib import Path
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from stage4_config import (TASKS, EMBED_MODEL, INDEX_DIR, OUT_DIR,
                           K_SHOTS)

MODEL = "claude-sonnet-5"
BATCH_DIR = OUT_DIR / "tiebreak_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Prompt construction (blind: no teacher labels)
# ----------------------------------------------------------------------
def build_prompt(cfg, examples, target):
    lines = [f"You are an expert in {cfg['task_name']} classification.",
             f"{cfg['task_name'].capitalize()} is {cfg['definition']}",
             f"Your task is to classify each sentence as one of: "
             f"{', '.join(cfg['labels'])}.",
             f"Respond with only the label, no explanation, no preamble."]
    for txt, lab in examples:
        lines.append(f"Sentence: {txt}")
        lines.append(f"Label: {lab}")
    lines.append(f"Sentence: {target}")
    lines.append("Label:")
    return "\n".join(lines)


SYSTEM = ("You are a classification model. Respond with exactly one label "
          "and nothing else. No preamble, no explanation, no analysis.")


def parse_label(raw, task):
    up = raw.upper().strip()
    if task == "sexism":
        if "NOT" in up and ("SEX" in up or "SEXI" in up):
            return "NOT SEXIST"
        if "SEXIST" in up or "SEXI" in up or up.startswith("SEX"):
            return "SEXIST"
        return None
    if task == "lexical_bias":
        if "NOT" in up and "BIAS" in up:
            return "NOT BIASED"
        if "BIAS" in up:
            return "BIASED"
        return None
    if task == "hate":
        if "NEITHER" in up or "NONE" in up:
            return "NEITHER"
        if "HATE" in up:
            return "HATE SPEECH"
        if "OFFENSIVE" in up or "OFFEN" in up:
            return "OFFENSIVE"
        return None
    if task == "sentiment":
        if "POSITIVE" in up or up.startswith("POS"):
            return "POSITIVE"
        if "NEGATIVE" in up or up.startswith("NEG"):
            return "NEGATIVE"
        if "NEUTRAL" in up or up.startswith("NEU"):
            return "NEUTRAL"
        return None
    return None


# ----------------------------------------------------------------------
# Build the set of disagreements + their prompts
# ----------------------------------------------------------------------
def load_disagreements(task, cfg):
    m = pl.read_parquet(OUT_DIR / f"{task}_mistral.parquet")
    g = pl.read_parquet(OUT_DIR / f"{task}_gemma.parquet")
    joined = (m.select(["id", "text_clean", "mistral_label"])
                .join(g.select(["id", "gemma_label"]), on="id", how="inner"))
    dis = joined.filter(
        pl.col("mistral_label").is_not_null()
        & pl.col("gemma_label").is_not_null()
        & (pl.col("mistral_label") != pl.col("gemma_label"))
    )
    print(f"[{task}] disagreements: {dis.height:,}")
    return dis


def build_requests(task, cfg, dis):
    """Return list of Anthropic batch request dicts."""
    # Retrieval index (same as teachers)
    train = pl.read_parquet(INDEX_DIR / f"{task}_gold_train.parquet")
    train_txt = train[cfg["gold_text_col"]].to_list()
    train_lab = train["label_str"].to_list()
    train_emb = np.load(INDEX_DIR / f"{task}_train_emb.npy")
    embedder = SentenceTransformer(EMBED_MODEL)

    ids = dis["id"].to_list()
    texts = dis["text_clean"].to_list()
    print(f"[{task}] embedding {len(texts):,} targets for retrieval...")
    q_emb = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True).astype(np.float32)

    requests = []
    for j, txt in enumerate(texts):
        sims = train_emb @ q_emb[j]
        top = np.argsort(-sims)[:K_SHOTS]
        ex = [(train_txt[k], train_lab[k]) for k in top]
        prompt = build_prompt(cfg, ex, txt)
        requests.append({
            "custom_id": str(ids[j]),
            "params": {
                "model": MODEL,
                "max_tokens": 8,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            },
        })
    return requests


# ----------------------------------------------------------------------
# Submit / collect
# ----------------------------------------------------------------------
def submit(task, cfg, client):
    dis = load_disagreements(task, cfg)
    if dis.height == 0:
        print(f"[{task}] no disagreements — nothing to tiebreak.")
        return
    requests = build_requests(task, cfg, dis)

    print(f"[{task}] submitting batch of {len(requests):,} requests...")
    batch = client.messages.batches.create(requests=requests)

    meta = {"batch_id": batch.id, "task": task, "n": len(requests)}
    (BATCH_DIR / f"{task}_batch.json").write_text(json.dumps(meta, indent=2))
    print(f"[{task}] batch_id = {batch.id}  (saved to {task}_batch.json)")
    return batch.id


def collect(task, cfg, client, poll_every=30):
    meta_path = BATCH_DIR / f"{task}_batch.json"
    if not meta_path.exists():
        raise SystemExit(f"No batch metadata for {task} — run --submit first.")
    meta = json.loads(meta_path.read_text())
    batch_id = meta["batch_id"]

    # Poll until ended
    while True:
        b = client.messages.batches.retrieve(batch_id)
        st = b.processing_status
        counts = b.request_counts
        print(f"[{task}] status={st} | done={counts.succeeded} "
              f"errored={counts.errored} processing={counts.processing}",
              flush=True)
        if st == "ended":
            break
        time.sleep(poll_every)

    # Stream results
    rows = []
    fails = 0
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            # Sonnet 5 may return thinking blocks; grab the TextBlock only.
            raw = ""
            for block in result.result.message.content:
                if getattr(block, "type", None) == "text":
                    raw = block.text.strip()
                    break
            lab = parse_label(raw, task) if raw else None
            if lab is None:
                fails += 1
        else:
            raw = f"__{result.result.type}__"
            lab = None
            fails += 1
        rows.append({"id": cid, "claude_label": lab, "claude_raw": raw})

    out = pl.DataFrame(rows)
    # cast id back to match the pool id dtype if needed
    out_path = OUT_DIR / f"{task}_tiebreak.parquet"
    out.write_parquet(out_path)
    print(f"[{task}] wrote {out.height:,} tiebreak labels -> {out_path}")
    print(f"[{task}] parse failures: {fails}")

    # Usage / cost report
    try:
        b = client.messages.batches.retrieve(batch_id)
        print(f"[{task}] batch ended; check console for exact token usage.")
    except Exception:
        pass


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="submit then block-poll until done")
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    cfg = TASKS[args.task]
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY in the environment.")
    client = Anthropic(api_key=api_key)

    if args.run:
        submit(args.task, cfg, client)
        collect(args.task, cfg, client, poll_every=args.poll)
    elif args.submit:
        submit(args.task, cfg, client)
    elif args.collect:
        collect(args.task, cfg, client, poll_every=args.poll)
    else:
        raise SystemExit("Pass one of --submit / --collect / --run")


if __name__ == "__main__":
    main()
