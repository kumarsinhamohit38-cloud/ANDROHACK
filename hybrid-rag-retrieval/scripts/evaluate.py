"""
scripts/evaluate.py - Run evaluation comparing Dense, BM25, and Hybrid retrieval.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --queries 100 --k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EVAL_DIR, FINAL_TOP_K
from src.data_loader import load_indexed_ms_marco_queries, load_ms_marco_queries
from src.bm25_search import BM25Retriever
from src.dense_search import DenseRetriever
from src.hybrid_search import HybridRetriever
from src.vector_store import VectorStore
from src.embeddings import get_embedding_model
from src.evaluation import run_evaluation
from src.utils import get_logger

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods.")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of evaluation queries to use.")
    parser.add_argument("--k", type=int, default=FINAL_TOP_K,
                        help="Evaluation cutoff k (top-k).")
    parser.add_argument(
        "--query-source",
        choices=["indexed", "validation"],
        default="indexed",
        help=(
            "'indexed' evaluates against relevance labels in the local corpus "
            "(default); 'validation' uses the separate MS MARCO validation split."
        ),
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Retrieval Evaluation (k={args.k}, queries={args.queries})")
    print(f"{'='*60}\n")

    # Load indexes
    print("Loading indexes ...")
    try:
        embedding_model = get_embedding_model()
        vector_store    = VectorStore()
        vector_store.create_collection()

        dense_retriever = DenseRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
        )

        bm25_retriever = BM25Retriever()
        bm25_retriever.load()

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            vector_store=vector_store,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load indexes: {e}")
        print("Please run:  python scripts/build_indexes.py")
        sys.exit(1)

    # Load evaluation queries
    print(f"Loading {args.queries} {args.query_source} evaluation queries ...")
    try:
        if args.query_source == "indexed":
            queries = load_indexed_ms_marco_queries(size=args.queries)
        else:
            queries = load_ms_marco_queries(size=args.queries)
    except Exception as e:
        print(f"[ERROR] Failed to load queries: {e}")
        sys.exit(1)

    if not queries:
        print("[ERROR] No evaluation queries loaded.")
        sys.exit(1)

    print(f"Loaded {len(queries)} queries.\n")

    # Run evaluation
    results = run_evaluation(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        queries=queries,
        k=args.k,
        max_queries=args.queries,
    )

    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    print(results["table"])

    out_path = Path(EVAL_DIR) / "evaluation_results.json"
    print(f"\n[OK] Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
