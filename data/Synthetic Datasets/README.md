# Dehydrated Tweet Dataset

This repository contains the **Tweet identifiers** and **our derived annotations**
for the four classification tasks used in this study: hate speech, sexism, lexical bias, and sentiment.

In line with the platform terms, we cannot release tweet texts or user information, but a copy of all the tweets we used can be found in the Internet Archive ({https://archive.org/details/twitterstream). Is archive used to be publicly available but current conditions for access depend on the Internet Archive.

## Files

| File | Task |
|------|------|
| `hate_tweet_ids.csv`         | Hate speech |
| `sexism_tweet_ids.csv`       | Sexism |
| `lexical_bias_tweet_ids.csv` | Lexical bias |
| `sentiment_tweet_ids.csv`    | Sentiment |

### Columns

| Column | Description |
|--------|-------------|
| `id`             | Tweet ID (string). Use this to rehydrate the tweet text. |
| `mistral_label`  | Label assigned by the Mistral teacher model. |
| `gemma_label`    | Label assigned by the Gemma teacher model. |
| `claude_label`   | Label assigned by the Claude tiebreaker (present only when the two teachers disagreed; otherwise `null`). |
| `final_label`    | The final label used in the study. |
| `label_source`   | Provenance of `final_label`: `teacher_agree` (both teachers agreed) or the tiebreaker source (teachers disagreed → Claude decided). |

> **Note:** `id` is stored as a **string** — do not import it as a numeric type,
> as Tweet IDs exceed 2^53 and will be corrupted (e.g. rounded/scientific
> notation) if parsed as integers or floats.