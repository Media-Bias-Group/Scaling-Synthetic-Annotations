# Scaling-Synthetic-Annotations
This repository contains data and code for the paper: "Conditional Scaling of Synthetic Annotations for Media Bias Classification"

Abstract tbd.

> **Data notice.**  Tweet texts and
> user information are **not** included. We release **Tweet IDs plus our
> annotations** for rehydration (see [`data/Synthetic Datasets/README.md`](data/Synthetic%20Datasets/README.md)). Furthermore Gold and Benchmark Datasets are publicly available and need to be downloaded from their respective sources for replication.

## Repository structure

```
Scaling-Synthetic-Annotations/
├── README.md
├── data/
│   ├── Synthetic Datasets/         
│   │   ├── README.md               
│   │   ├── *_tweet_ids.csv          # hate / sexism / sentiment / lexical_bias Tweet IDs + labels
│   │   └── *_news_final.parquet     # news-domain synthetic annotations (text included, public source)
│   └── benchmarks/                  # scripts to prepare external gold set and OOD benchmarks
├── src/
│   ├── Twitter pipeline/            # end-to-end Twitter data pipeline
│   │   ├── Corpus Construction/                     # extract → dedupe → subsample → pools
│   │   └── Retrieval-Augmented Multi-Teacher Annotation/   # index → annotate → tiebreak → finalize
│   ├── News Pipeline/               # parallel pipeline for the news domain
│   │   ├── Corpus Construction/
│   │   └── Retrieval-Augmented Multi-Teacher Annotation/
│   ├── Student Fine-Tuning and Evaluation/          # student training & evaluation
│   ├── Artificial Curation vs Natural Diversity/    # mixed vs. single-source split construction
│   ├── Baseline Viability/                          # LLM-baselines & human-supervised comparisons
│   └── Architectural Robustness/                    # RoBERTa / ModernBERT / DeBERTa-large ablation
├── results/                         # aggregated summaries + per-fold raw results
│   ├── stage5_summary.csv                            # main scaling results
│   ├── roberta_modenbert_summary.csv                 # cross-architecture ablation
│   ├── deberta_large_summary.csv                     # within-family scale ablation
│   ├── baseline viability_comparison.csv
│   ├── Artificial Curation vs Natural Diversity_summary.csv
│   └── stage5_results by fold/                        # per-(task, source, size, fold) run JSONs
└── analysis/                        # figure/table generation from the summaries
    ├── Results.py
    ├── twitter_scaling_analysis.py
    ├── baseline viability comparison.py
    └── human_supervised comparison.py
```


## Reproducing the experiments

> **Note on environments.** The DeBERTa-v3 experiments require
> `transformers==4.44.2` (later versions reintroduce a majority-class collapse),
> while the ModernBERT architecture ablation requires `transformers>=4.48`. Use
> **two separate environments** accordingly.

1. Prepare the corpora and benchmarks (`src/*/Corpus Construction/`,
   `data/benchmarks/prepare_*.py`).
2. Run annotation (`src/*/Retrieval-Augmented Multi-Teacher Annotation/`).
3. Fine-tune and evaluate students (`src/Student Fine-Tuning and Evaluation/`) and
   the ablation studies (`src/Baseline Viability/`, `src/Architectural
   Robustness/`).
4. Aggregate and plot (`analysis/`).

## Citation

If you use this code or data, please cite:

```bibtex
@article{[your_key],
  title   = {[Paper Title]},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2026},
}
```

## License

tbd
