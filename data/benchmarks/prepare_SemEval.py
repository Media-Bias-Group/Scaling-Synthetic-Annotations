import uuid
import html
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

tqdm.pandas()


def ultimately_unescape(text):
    if not isinstance(text, str):
        return text
    prev = None
    while prev != text:
        prev = text
        text = html.unescape(text)
    return text


def preprocess(text):
    text = ultimately_unescape(text)
    text = text.strip()
    return text


# ------------------------------------------------------------------
# 1. Loads the TweetEval sentiment dataset (SemEval-2017 Task 4 based)
# ------------------------------------------------------------------
ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", trust_remote_code=True)
print(ds)

# ------------------------------------------------------------------
# 2. Converts splits to DataFrames and tag them
# ------------------------------------------------------------------
df1 = ds["train"].to_pandas();       df1["split"] = "train"
df2 = ds["test"].to_pandas();        df2["split"] = "test"
df3 = ds["validation"].to_pandas();  df3["split"] = "validation"

df = pd.concat([df1, df2, df3], ignore_index=True)

# ------------------------------------------------------------------
# 3. Adds a unique id per row
# ------------------------------------------------------------------
df["text_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

# ------------------------------------------------------------------
# 4. Preprocesses text (exclude the test set to keep it untouched)
# ------------------------------------------------------------------
mask = df["split"] != "test"
df.loc[mask, "text"] = df.loc[mask, "text"].progress_apply(preprocess)

# ------------------------------------------------------------------
# 5. Maps numeric labels -> text labels
# ------------------------------------------------------------------
id2text = {0: "negative", 1: "neutral", 2: "positive"}
df["label_text"] = df["label"].apply(lambda x: id2text[x])

# ------------------------------------------------------------------
# 6. Saves the result
# ------------------------------------------------------------------
out_cols = ["text", "label", "label_text", "split", "text_id"]
df = df[out_cols]

df.to_csv("tweet_eval_sentiment.tsv", sep="\t", index=False)
# df.to_parquet("tweet_eval_sentiment.parquet", index=False)  # optional

print(f"\nSaved {len(df)} rows to tweet_eval_sentiment.tsv")
print(df["label_text"].value_counts())
print(df.head())
