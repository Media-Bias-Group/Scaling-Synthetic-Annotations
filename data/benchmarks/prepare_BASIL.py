"""BASIL prep v2 — match by phrase TEXT, not char offsets (offset base unreliable).
A sentence is BIASED if it contains the 'txt' of >=1 span whose bias type is in
POSITIVE_BIAS_TYPES ('lex'). Fully held out -> ALL rows.
"""
import json, re
import polars as pl
from pathlib import Path

REPO = Path("~/Gold Datasets/BASIL-main").expanduser()
OUT = (Path("~/data/stage5_benchmarks").expanduser()
       / "lexical_bias_benchmark.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

POSITIVE_BIAS_TYPES = {"lex"}

def flatten_sentences(article):
    sents = []
    for para in article.get("body-paragraphs", []):
        if isinstance(para, list):
            sents.extend(s for s in para if isinstance(s, str) and s.strip())
        elif isinstance(para, str) and para.strip():
            sents.append(para)
    return sents

def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def main():
    ann = {json.loads(p.read_text())["uuid"]: json.loads(p.read_text())
           for p in REPO.rglob("annotations/**/*.json")}
    art = {json.loads(p.read_text())["uuid"]: json.loads(p.read_text())
           for p in REPO.rglob("articles/**/*.json")}
    print(f"annotations: {len(ann)} | articles: {len(art)}")

    rows, total, matched, empty_txt = [], 0, 0, 0
    for uuid, a in art.items():
        sents = flatten_sentences(a)
        norm_sents = [norm(s) for s in sents]
        biased = set()
        for span in ann.get(uuid, {}).get("phrase-level-annotations", []):
            if span.get("bias") not in POSITIVE_BIAS_TYPES:
                continue
            total += 1
            txt = norm(span.get("txt", ""))
            if not txt:
                empty_txt += 1
                continue
            hit = next((i for i, ns in enumerate(norm_sents) if txt in ns), None)
            if hit is not None:
                biased.add(hit); matched += 1
        for i, s in enumerate(sents):
            rows.append({"id": f"{uuid}__s{i}", "text": s,
                         "label_str": "BIASED" if i in biased else "NOT BIASED"})

    df = pl.DataFrame(rows).filter(pl.col("text").str.strip_chars().str.len_chars() > 0)
    print(f"\nlex spans: {total} | matched to a sentence: {matched} "
          f"({100*matched/max(total,1):.1f}%) | empty txt: {empty_txt}")
    print(f"sentences: {df.height:,}")
    print(df.group_by("label_str").len().sort("len", descending=True))
    df.write_parquet(OUT, compression="zstd")
    print(f"Wrote -> {OUT}")

if __name__ == "__main__":
    main()
