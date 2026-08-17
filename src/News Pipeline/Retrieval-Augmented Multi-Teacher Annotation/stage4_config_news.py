"""Stage 4 config (NEWS) — lexbias + sentiment only."""
import os
from pathlib import Path

ROOT = Path(os.path.expanduser("~/"))
POOLS = ROOT / "stage3_pools"
OUT_DIR = ROOT / "stage4_annotations"
INDEX_DIR = Path(os.path.expanduser("~/data/stage4_index"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"   # identical to twitter [[14]]
K_SHOTS = 8
SPLIT_SEED = 42
TRAIN_FRAC = 0.8
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"   # [[14]]
TEACHERS = {
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",  # identical model ids [[14]]
    "gemma":   "google/gemma-3-27b-it",
}

TASKS = {
    "lexical_bias": {
        "task_name": "lexical bias",
        "definition": ("a pattern of using certain words that reflects a particular way "
                       "of thinking about a group or an individual based on their social category"),
        "labels": ["BIASED", "NOT BIASED"],
        "pool_file": os.path.join(POOLS, "news_lexical_bias_pool_balanced.parquet"),
        "gold_text_col": "text",
        # SAME gold as twitter — gold sources don't change with pool domain [[15]]
        "gold_file": Path(os.path.expanduser("~/twitter_pipeline")) / "data" / "gold" / "lexical_bias_gold.parquet",
    },
    "sentiment": {
        "task_name": "sentiment",
        "definition": ("a feeling, view, or opinion that is held or expressed. Positive sentiment "
                       "indicates approval or happiness, negative sentiment expresses dissatisfaction "
                       "or dislike, and neutral sentiment conveys objective information without "
                       "expressing strong emotions"),
        "labels": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "pool_file": os.path.join(POOLS, "news_sentiment_pool_balanced.parquet"),
        "gold_text_col": "text",
        "gold_file": Path(os.path.expanduser("~/twitter_pipeline")) / "data" / "gold" / "sentiment_gold.parquet",
    },
}
