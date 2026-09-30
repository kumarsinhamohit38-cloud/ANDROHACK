"""
scripts/preprocess_data.py - Standalone preprocessing script.

This is called automatically by download_data.py.
You can also run it manually if you want to re-preprocess without
re-downloading.

Usage:
    python scripts/preprocess_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, RAW_DATA_DIR
from src.preprocessing import preprocess_passages
from src.utils import get_logger, load_jsonl, save_jsonl

logger = get_logger("preprocess_data")


def main():
    raw_path = RAW_DATA_DIR / "passages_raw.jsonl"
    out_path = PROCESSED_DIR / "passages.jsonl"

    if not raw_path.exists():
        print(f"[ERROR] Raw data not found: {raw_path}")
        print("Please run:  python scripts/download_data.py")
        return

    print(f"Loading raw passages from {raw_path} ...")
    raw = load_jsonl(raw_path)

    print(f"Preprocessing {len(raw)} passages ...")
    processed = preprocess_passages(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(processed, out_path)
    print(f"[OK] Saved {len(processed)} processed passages to {out_path}")


if __name__ == "__main__":
    main()
