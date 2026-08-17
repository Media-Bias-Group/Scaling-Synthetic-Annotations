"""inspect_newsmediabias.py — Inspect vector-institute/newsmediabias-plus
for usability as a third source domain in the SA-FT pipeline.

Answers:
  1. Size & splits — can we reach 128k/task after sentence-splitting + filtering?
  2. Field names & types — what's the text field, are there native bias labels?
  3. Text granularity — article-level? How long, and how many sentences per record?
  4. Native label schema — does it map to any of our 4 tasks (only lexical_bias likely)?
"""
from datasets import load_dataset
import numpy as np
import re

# ----------------------------------------------------------------------
# 1. Load + basic structure
# ----------------------------------------------------------------------

ds = load_dataset("vector-institute/newsmediabias-plus")
print("=" * 70)
print("SPLITS & SIZES")
print("=" * 70)
for split in ds:
    print(f"  {split}: {len(ds[split]):,} records")

train = ds[list(ds.keys())[0]]  # usually 'train'

print("\n" + "=" * 70)
print("FEATURES / SCHEMA")
print("=" * 70)
for name, feat in train.features.items():
    print(f"  {name:35s} {feat}")

# ----------------------------------------------------------------------
# 2. First record (full) + a few samples (truncated text)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("FIRST RECORD (full)")
print("=" * 70)
rec0 = train[0]
for k, v in rec0.items():
    s = str(v)
    print(f"\n[{k}]")
    print(s[:800] + (" ...[truncated]" if len(s) > 800 else ""))

# ----------------------------------------------------------------------
# 3. Identify the main text field automatically (longest string field)
# ----------------------------------------------------------------------
str_fields = [k for k, v in rec0.items() if isinstance(v, str)]
sample = train[:200]
avg_len = {}
for f in str_fields:
    lens = [len(x) for x in sample[f] if isinstance(x, str)]
    avg_len[f] = np.mean(lens) if lens else 0
text_field = max(avg_len, key=avg_len.get) if avg_len else None

print("\n" + "=" * 70)
print("STRING FIELD AVG CHAR LENGTHS (first 200 records)")
print("=" * 70)
for f, l in sorted(avg_len.items(), key=lambda x: -x[1]):
    print(f"  {f:35s} {l:8.0f} chars")
print(f"\n  --> Inferred main text field: '{text_field}'")

# ----------------------------------------------------------------------
# 4. Granularity check — how long are texts, how many sentences each
# ----------------------------------------------------------------------
def naive_sentences(t):
    return [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if len(s.split()) >= 5]

n = min(2000, len(train))
subset = train[:n]
char_lens, word_lens, sent_counts = [], [], []
for t in subset[text_field]:
    if not isinstance(t, str):
        continue
    char_lens.append(len(t))
    word_lens.append(len(t.split()))
    sent_counts.append(len(naive_sentences(t)))

char_lens, word_lens, sent_counts = map(np.array, (char_lens, word_lens, sent_counts))

print("\n" + "=" * 70)
print(f"GRANULARITY (main text field, first {n} records)")
print("=" * 70)
for name, arr in [("chars", char_lens), ("words", word_lens),
                  ("sentences(>=5w)", sent_counts)]:
    print(f"  {name:18s} mean={arr.mean():8.1f}  median={np.median(arr):8.1f}  "
          f"p95={np.percentile(arr,95):8.1f}  max={arr.max():8.0f}")

total_articles = len(train)
mean_sents = sent_counts.mean()
print(f"\n  Articles: {total_articles:,}")
print(f"  Est. usable sentences (>=5 words): ~{int(total_articles * mean_sents):,}")
print(f"  --> 128k/task feasible from sentences? "
      f"{'YES' if total_articles * mean_sents > 128_000 * 4 else 'CHECK — may be tight'}")

# ----------------------------------------------------------------------
# 5. Native label schema
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("CANDIDATE LABEL FIELDS (value counts, first 2000)")
print("=" * 70)
label_like = [k for k, v in rec0.items()
              if k != text_field and (
                  isinstance(v, (int, bool)) or
                  (isinstance(v, str) and len(str(v)) < 60) or
                  isinstance(v, list))]
for f in label_like:
    vals = subset[f]
    # Flatten simple cases; show top categories
    try:
        flat = [str(x) for x in vals if x is not None]
        uniq, cnts = np.unique(flat, return_counts=True)
        order = np.argsort(-cnts)[:8]
        print(f"\n  [{f}]  ({len(uniq)} unique)")
        for i in order:
            print(f"      {uniq[i][:50]:50s} {cnts[i]}")
    except Exception as e:
        print(f"\n  [{f}] (could not summarize: {e})")
