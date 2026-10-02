# Data Directory

This directory holds all dataset files. Put each topic's JSONL file in `raw/` (subfolders are supported). The preprocessing script combines every JSONL file it finds into one searchable corpus while preserving each record's dataset label.

## Structure

```
data/
├── raw/         ← Topic JSONL files and raw downloads (gitignored)
└── processed/   ← Combined, preprocessed passages.jsonl (auto-created, gitignored)
```

## Files

### `processed/passages.jsonl`

Created by `python scripts/preprocess_data.py` from every `*.jsonl` file under `data/raw/`.

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

Each record must have a non-empty `text` field. A recommended record format is:

```json
{"id":"unique-id","text":"Passage text","metadata":{"source":"curated_demo","dataset":"Python (demo samples)","domain":"Programming"}}
```

Keep one JSON object per line. Files may be separated by topic, for example `raw/python.jsonl` and `raw/topics/fiqa.jsonl`. The app's dataset filter uses `metadata.dataset`.

## MS MARCO

**MS MARCO** (Microsoft MAchine Reading COmprehension)

- Source: HuggingFace `ms_marco` dataset
- Version: v1.1
- Train split: ~808K queries, ~8M passages
- Validation split: ~101K queries (used for evaluation)

## Download

```bash
# Optional: download MS MARCO demo data (10,000 passages)
python scripts/download_data.py --mode demo

# Optional: download a larger MS MARCO sample
python scripts/download_data.py --mode full

# Combine all topic files, then rebuild search indexes
python scripts/preprocess_data.py
python scripts/build_indexes.py --recreate
```

The 20 topic files under `raw/topics/` currently contain 50 curated demo passages each. These examples are not records copied from the named benchmark datasets. Download official benchmark data separately and follow each dataset's license/terms before using it. The Python file remains separate and is included in the combined corpus.
