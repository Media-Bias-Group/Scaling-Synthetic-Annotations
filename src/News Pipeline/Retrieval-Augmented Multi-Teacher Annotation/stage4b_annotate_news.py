"""Resumable teacher annotation via DeepInfra.
Concurrent requests, deterministic order, skips already-done ids, incremental flush."""
import argparse, os, time, numpy as np, polars as pl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from stage4_config_news import (TASKS, TEACHERS, EMBED_MODEL, INDEX_DIR,
                           OUT_DIR, K_SHOTS, DEEPINFRA_BASE_URL)


# ----------------------------------------------------------------------
# Single argument parser (source of truth)
# ----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="sexism", choices=list(TASKS))
ap.add_argument("--teacher", required=True, choices=list(TEACHERS))
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--flush", type=int, default=500)
ap.add_argument("--workers", type=int, default=32, help="Concurrent API workers")
args = ap.parse_args()

TASK = args.task
cfg = TASKS[TASK]


def build_prompt(examples, target):
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


def parse_label(raw, task):
    """Task-specific label extraction from raw model output."""
    up = raw.upper().strip()

    if task == "sexism":
        # Check NOT SEXIST first (before SEXIST substring match)
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


def _flush(path, rows):
    if not rows:
        return
    new = pl.DataFrame(rows)
    if Path(path).exists():
        old = pl.read_parquet(path)
        pl.concat([old, new], how="vertical_relaxed").write_parquet(
            path, compression="zstd")
    else:
        new.write_parquet(path, compression="zstd")


def classify_one(client, model_id, prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content":
                     "You are a classification model. Respond with exactly one label "
                     "and nothing else. No preamble, no explanation, no analysis."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8,
                temperature=0.0,
                timeout=60,
            )
            raw = r.choices[0].message.content.strip()
            in_tok = r.usage.prompt_tokens
            out_tok = r.usage.completion_tokens
            return raw, in_tok, out_tok, None
        except Exception as e:
            if attempt == max_retries - 1:
                return "", 0, 0, str(e)
            time.sleep(2 ** attempt)
    return "", 0, 0, "max_retries_exceeded"


def main():
    out_file = OUT_DIR / f"{TASK}_{args.teacher}.parquet"
    model_id = TEACHERS[args.teacher]

    api_key = os.environ.get("DEEPINFRA_TOKEN")
    if not api_key:
        raise SystemExit("Set DEEPINFRA_TOKEN in the environment.")
    client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)

    print(f"Task: {TASK}")
    print(f"Model: {model_id}")
    print(f"Endpoint: {DEEPINFRA_BASE_URL}")
    print(f"Concurrent workers: {args.workers}")

    pool = pl.read_parquet(cfg["pool_file"]).sort("id")
    if args.limit:
        pool = pool.head(args.limit)

    done_ids = set()
    if out_file.exists():
        done_ids = set(pl.read_parquet(out_file, columns=["id"])["id"].to_list())
        print(f"Resuming: {len(done_ids):,} already done")
    todo = pool.filter(~pl.col("id").is_in(list(done_ids)))
    print(f"To annotate: {todo.height:,} / {pool.height:,}")
    if todo.height == 0:
        print("Nothing to do.")
        return

    # Retrieval index
    train = pl.read_parquet(INDEX_DIR / f"{TASK}_gold_train.parquet")
    train_txt = train[cfg["gold_text_col"]].to_list()
    train_lab = train["label_str"].to_list()
    train_emb = np.load(INDEX_DIR / f"{TASK}_train_emb.npy")
    embedder = SentenceTransformer(EMBED_MODEL)

    ids = todo["id"].to_list()
    texts = todo["text_clean"].to_list()
    print("Embedding targets for retrieval...")
    q_emb = embedder.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True).astype(np.float32)

    # Build all prompts upfront
    print("Building prompts...")
    prompts = []
    for j, txt in enumerate(texts):
        sims = train_emb @ q_emb[j]
        top = np.argsort(-sims)[:K_SHOTS]
        ex = [(train_txt[k], train_lab[k]) for k in top]
        prompts.append(build_prompt(ex, txt))

    print(f"Dispatching {len(prompts):,} requests with {args.workers} workers...")
    rows, tot_in, tot_out, fails, errors = [], 0, 0, 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {
            pool_exec.submit(classify_one, client, model_id, prompts[i]): i
            for i in range(len(prompts))
        }

        for done_count, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            raw, in_tok, out_tok, err = fut.result()

            if err:
                errors += 1
                lab = None
            else:
                lab = parse_label(raw, TASK)
                if lab is None:
                    fails += 1

            rows.append({"id": ids[i], "text_clean": texts[i],
                         f"{args.teacher}_label": lab,
                         f"{args.teacher}_raw": raw})
            tot_in += in_tok
            tot_out += out_tok

            if done_count % args.flush == 0 or done_count == len(prompts):
                _flush(out_file, rows)
                rows = []
                el = time.time() - t0
                print(f"  {done_count:,}/{len(prompts):,} | "
                      f"{done_count/el:.1f} inst/s | "
                      f"in {tot_in:,} out {tot_out:,} tok | "
                      f"parse-fail {fails} | errors {errors}",
                      flush=True)

    if rows:
        _flush(out_file, rows)

    el = time.time() - t0
    price_map = {
        "mistralai/Mistral-Small-24B-Instruct-2501": (0.07, 0.14),
        "google/gemma-3-27b-it": (0.09, 0.17),
    }
    p_in, p_out = price_map.get(model_id, (0.0, 0.0))
    cost = tot_in / 1e6 * p_in + tot_out / 1e6 * p_out

    print("=" * 60)
    print(f"Task: {TASK}   Teacher: {args.teacher} ({model_id})")
    print(f"Annotated: {len(prompts):,} in {el/60:.1f} min "
          f"({len(prompts)/el:.1f} inst/s)")
    print(f"Tokens: in {tot_in:,}  out {tot_out:,}")
    print(f"Parse failures: {fails}   API errors: {errors}")
    print(f"Estimated cost: ${cost:.2f}")
    print(f"Output: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
