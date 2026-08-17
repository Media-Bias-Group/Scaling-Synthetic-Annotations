import os, re, hashlib
import polars as pl
from datasets import load_dataset

HF_TOKEN = os.environ["HF_TOKEN"]
ds = load_dataset("vector-institute/newsmediabias-plus", split="train", token=HF_TOKEN)

def split_sentences(t):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t.strip())
            if len(s.split()) >= 5]   # mirror your >=5-word / len>=20 filter

rows = []
for rec in ds:
    body = (rec.get("article_text") or "").strip()
    for sent in split_sentences(body):
        # stable id: article id + sentence hash (so re-runs are reproducible)
        sid = hashlib.md5(f"{rec['unique_id']}::{sent}".encode()).hexdigest()[:16]
        rows.append({
            "id": f"news_{sid}",              # 'news_' prefix keeps ids disjoint from tweets
            "text_clean": sent,
            "article_id": rec["unique_id"],
            "outlet": rec.get("outlet"),
            "nlp_label": rec.get("nlp_label"),  # weak prior, lexbias only
        })

df = pl.DataFrame(rows).unique(subset=["text_clean"], keep="first")  # dedup
df = df.filter(pl.col("text_clean").str.len_chars() >= 20)           # length filter [[13]]
df.write_parquet("~/data/stage3_pools/news_sentences.parquet")
