# Data Directory

This directory holds all dataset files.

## Structure

```
data/
├── raw/         ← Raw downloaded files (auto-created, gitignored)
└── processed/   ← Preprocessed passages.jsonl (auto-created, gitignored)
```

## Files

### `processed/passages.jsonl`

Created by `python scripts/download_data.py`.

Each line is a JSON object:

```json
{
    "id": "passage_0",
    "text": "Paris is the capital city of France...",
    "tokens": ["paris", "capital", "city", "france"],
    "metadata": {
        "source": "ms_marco",
        "dataset": "ms_marco",
        "query_id": "12345",
        "query_text": "What is the capital of France?",
        "is_selected": true,
        "url": "https://example.com",
        "passage_idx": 0,
        "text_length": 38,
        "token_count": 7
    }
}
```

## Dataset

**MS MARCO** (Microsoft MAchine Reading COmprehension)

- Source: HuggingFace `ms_marco` dataset
- Version: v1.1
- Train split: ~808K queries, ~8M passages
- Validation split: ~101K queries (used for evaluation)

## Download

```bash
# Demo mode (10,000 passages) – recommended for first run
python scripts/download_data.py --mode demo

# Full mode (100,000 passages)
python scripts/download_data.py --mode full
```
