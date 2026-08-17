"""Stage 3b (news): run per-task SOTA pre-filter classifiers on cleaned news
sentences, emitting <task>_pred / <task>_conf columns — the exact contract
that stage3c balancing consumes [[1]].
"""
import os, argparse
from pathlib import Path
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(os.path.expanduser("/workspace"))
IN  = ROOT / "stage2_files" / "news_sentences_clean.parquet"
OUTD = ROOT / "stage3_pools"; OUTD.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# --- Use the SAME model ids / label->int maps you used for twitter stage3b ---
# hate: 0=hate, 1=offensive, 2=normal  (matches stage3c hate_pred semantics [[1]])
TASK_MODELS = {
    "hate":         "cardiffnlp/twitter-roberta-base-hate-latest",   # example — use YOUR twitter choice
    "sexism":       "tum-nlp/bertweet-sexism",
    "lexical_bias": "mediabiasgroup/magpie-babe-ft",                  # BABE-style lexbias [[3]]
    "sentiment":    "cardiffnlp/twitter-roberta-base-sentiment-latest",
}

@torch.no_grad()
def classify_batch(texts, model, tok, batch_size=64):
    preds, confs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, max_length=256, padding=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        preds.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())
    return preds, confs

def run_task(task, df):
    model_id = TASK_MODELS[task]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval()
    texts = df["text_clean"].to_list()
    preds, confs = classify_batch(texts, model, tok)
    out = df.with_columns([
        pl.Series(f"{task}_pred", preds, dtype=pl.Int64),
        pl.Series(f"{task}_conf", confs, dtype=pl.Float64),
    ])
    out_path = OUTD / f"news_{task}_scored.parquet"
    out.write_parquet(out_path, compression="zstd", compression_level=3)
    print(f"[{task}] wrote {out.height:,} rows -> {out_path}")
    print(out.group_by(f"{task}_pred").len().sort(f"{task}_pred"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_MODELS), required=True)
    args = ap.parse_args()
    df = pl.read_parquet(IN)
    print(f"Classifying {df.height:,} news sentences for task={args.task} on {DEVICE}")
    run_task(args.task, df)

if __name__ == "__main__":
    main()
