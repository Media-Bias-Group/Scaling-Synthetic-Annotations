"""Stage 4 configuration — all annotation tasks."""
import os
from pathlib import Path

ROOT = Path(os.path.expanduser("~/"))
GOLD_DIR = ROOT / "Gold Datasets"
POOLS = ROOT / "data" / "stage3_pools"
GOLD_PARQUET_DIR = ROOT / "data" / "gold"          # pre-processed gold parquets
OUT_DIR = ROOT / "data" / "stage4_annotations"
INDEX_DIR = ROOT / "data" / "stage4_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
GOLD_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
K_SHOTS = 8
SPLIT_SEED = 42
TRAIN_FRAC = 0.8
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
TEACHERS = {
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
    "gemma":   "google/gemma-3-27b-it",
}

VLLM_URL = "http://localhost:8000/v1"

TASKS = {
    "sexism": {
        "task_name": "sexism",
        "definition": ("prejudice, stereotyping, or discrimination, typically "
                       "against women, on the basis of sex"),
        "labels": ["SEXIST", "NOT SEXIST"],
        "pool_file": os.path.join(POOLS, "sexism_pool.parquet"),
        "gold_text_col": "text",
        # CMSB gold (CSV, boolean sexist column -> label_map)
        "gold_file": GOLD_DIR / "cmsb" / "sexism_gold.csv",
        "gold_label_col": "sexist",
        "label_map": {True: "SEXIST", False: "NOT SEXIST"},
    },
    "hate": {
        "task_name": "hate speech and offensive language",
        "definition": (
            "hate speech is language that is used to express hatred towards a targeted group "
            "or is intended to be derogatory, to humiliate, or to insult the members of the group. "
            "offensive language are words or expressions that are generally vulgar, obscene, "
            "profane, or insulting but do not target individuals or groups on the basis of their characteristics"
        ),
        "labels": ["HATE SPEECH", "OFFENSIVE", "NEITHER"],
        "pool_file": os.path.join(POOLS, "hate_pool_balanced.parquet"),
        "gold_text_col": "text",
        # HateXplain gold — pre-processed parquet with label_str already set
        "gold_file": GOLD_PARQUET_DIR / "hate_gold.parquet",
    },
    "lexical_bias": {
        "task_name": "lexical bias",
        "definition": (
            "a pattern of using certain words that reflects a particular way of thinking "
            "about a group or an individual based on their social category"
        ),
        "labels": ["BIASED", "NOT BIASED"],
        "pool_file": os.path.join(POOLS, "lexbias_pool.parquet"),
        "gold_text_col": "text",
        # BABE gold — pre-processed parquet with label_str already set
        "gold_file": GOLD_PARQUET_DIR / "lexical_bias_gold.parquet",
    },
    "sentiment": {
        "task_name": "sentiment",
        "definition": (
            "a feeling, view, or opinion that is held or expressed. Positive sentiment indicates "
            "approval or happiness, negative sentiment expresses dissatisfaction or dislike, "
            "and neutral sentiment conveys objective information without expressing strong emotions"
        ),
        "labels": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "pool_file": os.path.join(POOLS, "sentiment_pool.parquet"),
        "gold_text_col": "text",
        # SemEval-2017 gold — pre-processed parquet with label_str already set
        "gold_file": GOLD_PARQUET_DIR / "sentiment_gold.parquet",
    },
}
