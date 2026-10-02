"""
scripts/preprocess_data.py - Standalone preprocessing script.

This is called automatically by download_data.py.
You can also run it manually if you want to re-preprocess without
re-downloading.

Usage:
    python scripts/preprocess_data.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, RAW_DATA_DIR
from src.preprocessing import preprocess_passages
from src.utils import get_logger, save_jsonl

logger = get_logger("preprocess_data")


def main():
    out_path = PROCESSED_DIR / "passages.jsonl"

    raw_files = sorted(RAW_DATA_DIR.rglob("*.jsonl"))
    if not raw_files:
        print(f"[ERROR] No JSONL datasets found under: {RAW_DATA_DIR}")
        print("Add JSONL files to data/raw (or a subfolder) and run this script again.")
        return

    raw = []
    seen_ids = set()
    for raw_path in raw_files:
        print(f"Loading {raw_path.relative_to(RAW_DATA_DIR)} ...")
        with raw_path.open("r", encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {raw_path.name}, line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict) or not str(record.get("text", "")).strip():
                    logger.warning("Skipping record without passage text: %s line %s", raw_path.name, line_number)
                    continue

                record = dict(record)
                metadata = record.get("metadata")
                record["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
                record["metadata"].setdefault("dataset", raw_path.stem)
                record["metadata"].setdefault("source", "user_upload")
                record["metadata"].setdefault("input_file", raw_path.stem)

                original_id = str(record.get("id") or f"row_{line_number}")
                record_id = original_id
                if record_id in seen_ids:
                    record_id = f"{raw_path.stem}_{original_id}"
                    suffix = 2
                    while record_id in seen_ids:
                        record_id = f"{raw_path.stem}_{original_id}_{suffix}"
                        suffix += 1
                record["id"] = record_id
                seen_ids.add(record_id)
                raw.append(record)

    if not raw:
        print("[ERROR] No usable records with non-empty 'text' fields were found.")
        return

    print(f"Preprocessing {len(raw)} passages from {len(raw_files)} JSONL files ...")
    processed = preprocess_passages(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(processed, out_path)
    print(f"[OK] Saved {len(processed)} combined passages to {out_path}")


if __name__ == "__main__":
    main()
