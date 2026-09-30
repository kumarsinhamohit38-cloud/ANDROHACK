"""
scripts/download_data.py - Download the MS MARCO dataset.

Run this ONCE before building indexes.

Usage:
    python scripts/download_data.py --mode demo
    python scripts/download_data.py --mode full --size 100000
"""

import argparse
import sys
from pathlib import Path

# Make 'src' importable from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEMO_SIZE, FULL_DATASET_SIZE, PROCESSED_DIR
from src.data_loader import load_ms_marco
from src.preprocessing import preprocess_passages
from src.utils import get_logger, save_jsonl

logger = get_logger("download_data")


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess MS MARCO passages."
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "full"],
        default="demo",
        help="'demo' loads ~10k passages (fast); 'full' loads 100k+ (slow).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Override the number of passages to download.",
    )
    args = parser.parse_args()

    size = args.size or (DEMO_SIZE if args.mode == "demo" else FULL_DATASET_SIZE)

    print(f"\n{'='*60}")
    print(f" Downloading MS MARCO ({args.mode} mode, {size} passages)")
    print(f"{'='*60}\n")
    print("This may take a few minutes on first run (downloading dataset).\n")

    # Step 1: Download raw passages
    passages = load_ms_marco(size=size, mode=args.mode)

    # Step 2: Preprocess
    processed = preprocess_passages(passages)

    # Step 3: Save to disk
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "passages.jsonl"
    save_jsonl(processed, out_path)

    print(f"\n[OK] Saved {len(processed)} passages to: {out_path}")
    print("\nNext step:")
    print("  python scripts/build_indexes.py")


if __name__ == "__main__":
    main()
