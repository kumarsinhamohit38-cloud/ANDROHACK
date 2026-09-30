"""
scripts/build_indexes.py - Build both the vector and BM25 indexes.

This is the ONE command you run after downloading the data.

Usage:
    python scripts/build_indexes.py
    python scripts/build_indexes.py --recreate   # rebuild from scratch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PASSAGES_FILE, BATCH_SIZE
from src.embeddings import EmbeddingModel, EMBEDDING_MODEL
from src.vector_store import VectorStore
from src.bm25_search import BM25Retriever
from src.utils import get_logger, load_jsonl

logger = get_logger("build_indexes")


def build_vector_index(passages: list[dict], recreate: bool = False) -> None:
    """Build the ChromaDB vector index."""
    print("\n-- Building vector index (ChromaDB) ----------------------")

    # Load embedding model
    model = EmbeddingModel(EMBEDDING_MODEL)
    model.load()

    # Get texts for embedding
    texts = [p["text"] for p in passages]
    print(f"Encoding {len(texts)} passages (batch_size={BATCH_SIZE}) ...")
    print("This may take several minutes on first run ...\n")

    embeddings = model.encode_documents(texts, batch_size=BATCH_SIZE, show_progress=True)

    # Insert into ChromaDB
    store = VectorStore()
    store.create_collection(recreate=recreate)
    store.insert(passages, embeddings)

    print(f"\n[OK] Vector index built. Total passages: {store.count()}")


def build_bm25_index(passages: list[dict]) -> None:
    """Build the BM25 index."""
    print("\n-- Building BM25 index ------------------------------------")

    retriever = BM25Retriever()
    retriever.build(passages)
    retriever.save()

    print("[OK] BM25 index built and saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Build vector and BM25 indexes from preprocessed passages."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild existing indexes from scratch.",
    )
    parser.add_argument(
        "--only",
        choices=["vector", "bm25"],
        default=None,
        help="Build only one type of index.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Building Indexes")
    print(f"{'='*60}")

    # Load preprocessed passages
    if not PASSAGES_FILE.exists():
        print(f"\n[ERROR] Passages file not found: {PASSAGES_FILE}")
        print("Please run first:  python scripts/download_data.py")
        sys.exit(1)

    print(f"\nLoading passages from {PASSAGES_FILE} ...")
    passages = load_jsonl(PASSAGES_FILE)
    print(f"Loaded {len(passages)} passages.")

    # Build indexes
    if args.only == "vector":
        build_vector_index(passages, recreate=args.recreate)
    elif args.only == "bm25":
        build_bm25_index(passages)
    else:
        build_vector_index(passages, recreate=args.recreate)
        build_bm25_index(passages)

    print(f"\n{'='*60}")
    print("  Indexes ready!  You can now run:  streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
