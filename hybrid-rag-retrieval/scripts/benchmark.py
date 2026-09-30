"""
scripts/benchmark.py - Measure query latency across all retrieval methods.

Runs N queries (default 50) and reports:
    mean, median, p50, p95, p99 latency

Results are saved to results/benchmarks/

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --queries 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BENCHMARK_DIR, BENCHMARK_QUERIES
from src.bm25_search import BM25Retriever
from src.dense_search import DenseRetriever
from src.hybrid_search import HybridRetriever
from src.vector_store import VectorStore
from src.embeddings import get_embedding_model
from src.utils import get_logger, save_json

logger = get_logger("benchmark")

# Sample queries for benchmarking
BENCHMARK_QUERY_LIST = [
    "What causes hallucinations in large language models?",
    "How does vector search work?",
    "What is BM25 retrieval?",
    "Define reciprocal rank fusion",
    "What is the capital of France?",
    "How does the immune system fight viruses?",
    "What are the symptoms of diabetes?",
    "How do neural networks learn?",
    "What is retrieval augmented generation?",
    "How does BERT work?",
    "What is fine-tuning in machine learning?",
    "What causes inflation?",
    "How do vaccines work?",
    "What is the speed of light?",
    "How does photosynthesis work?",
    "What is the Pythagorean theorem?",
    "How does encryption protect data?",
    "What is a transformer architecture?",
    "What are embeddings in NLP?",
    "How does cosine similarity work?",
    "What is gradient descent?",
    "How does overfitting happen?",
    "What is regularisation in machine learning?",
    "What causes climate change?",
    "What is quantum computing?",
    "How does a CPU work?",
    "What is the difference between RAM and ROM?",
    "How does the internet work?",
    "What is TCP/IP?",
    "How does DNS resolution work?",
    "What is a relational database?",
    "How does indexing speed up database queries?",
    "What is a binary search tree?",
    "What is the time complexity of quicksort?",
    "How does garbage collection work?",
    "What is object-oriented programming?",
    "What is a REST API?",
    "How does OAuth work?",
    "What is containerisation in software?",
    "How does Kubernetes orchestrate containers?",
    "What is a microservices architecture?",
    "What causes software bugs?",
    "How does a compiler work?",
    "What is machine learning bias?",
    "How is model accuracy measured?",
    "What is cross-validation?",
    "What is the ROC curve?",
    "How does backpropagation work?",
    "What is attention in transformers?",
    "What is zero-shot learning?",
]


def run_benchmark(retriever, queries: list[str], name: str) -> dict:
    """Run retrieval for each query and measure latency."""
    latencies = []

    print(f"\n  Benchmarking {name} ...")
    for i, query in enumerate(queries):
        start = time.perf_counter()
        try:
            if name == "hybrid":
                result = retriever.search(query)
            else:
                result = retriever.search(query, top_k=5)
        except Exception as e:
            logger.warning(f"Query {i} failed: {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(queries)} queries done ...")

    if not latencies:
        return {
            "name": name, "n": 0,
            "mean_ms": 0, "median_ms": 0,
            "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
        }

    stats = {
        "name":      name,
        "n":         len(latencies),
        "mean_ms":   round(float(np.mean(latencies)),   2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p50_ms":    round(float(np.percentile(latencies, 50)), 2),
        "p95_ms":    round(float(np.percentile(latencies, 95)), 2),
        "p99_ms":    round(float(np.percentile(latencies, 99)), 2),
        "min_ms":    round(float(np.min(latencies)), 2),
        "max_ms":    round(float(np.max(latencies)), 2),
        "raw_ms":    [round(x, 2) for x in latencies],
    }
    return stats


def print_stats(stats: dict) -> None:
    print(f"\n  {stats['name'].upper()} latency ({stats['n']} queries):")
    print(f"    Mean   : {stats['mean_ms']:>8.2f} ms")
    print(f"    Median : {stats['median_ms']:>8.2f} ms")
    print(f"    p50    : {stats['p50_ms']:>8.2f} ms")
    print(f"    p95    : {stats['p95_ms']:>8.2f} ms  {'PASS' if stats['p95_ms'] < 300 else 'WARN >300ms target'}")
    print(f"    p99    : {stats['p99_ms']:>8.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency.")
    parser.add_argument("--queries", type=int, default=BENCHMARK_QUERIES,
                        help="Number of benchmark queries.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Retrieval Latency Benchmark ({args.queries} queries)")
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

    print("Indexes loaded.")

    # Select benchmark queries
    queries = BENCHMARK_QUERY_LIST * (args.queries // len(BENCHMARK_QUERY_LIST) + 1)
    queries = queries[:args.queries]

    # Warm up (first query is slow due to JIT / cache misses)
    print("\nWarming up ...")
    try:
        dense_retriever.search("warm up query", top_k=5)
        bm25_retriever.search("warm up query", top_k=5)
        hybrid_retriever.search("warm up query")
    except Exception:
        pass

    # Run benchmarks
    dense_stats  = run_benchmark(dense_retriever,  queries, "dense")
    bm25_stats   = run_benchmark(bm25_retriever,   queries, "bm25")
    hybrid_stats = run_benchmark(hybrid_retriever, queries, "hybrid")

    # Print results
    print(f"\n{'='*60}")
    print("  Benchmark Results")
    print(f"{'='*60}")
    print_stats(dense_stats)
    print_stats(bm25_stats)
    print_stats(hybrid_stats)

    print(f"\n{'-'*60}")
    print(f"  {'Method':<10} | {'p95 (ms)':<10} | {'Target <300ms'}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*15}")
    for s in [dense_stats, bm25_stats, hybrid_stats]:
        target = "PASS" if s["p95_ms"] < 300 else "FAIL"
        print(f"  {s['name']:<10} | {s['p95_ms']:<10.2f} | {target}")
    print(f"{'-'*60}\n")

    # Save results
    all_stats = {
        "dense":  dense_stats,
        "bm25":   bm25_stats,
        "hybrid": hybrid_stats,
        "n_queries": args.queries,
    }
    out_dir = Path(BENCHMARK_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    save_json(all_stats, out_path)
    print(f"[OK] Benchmark results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
