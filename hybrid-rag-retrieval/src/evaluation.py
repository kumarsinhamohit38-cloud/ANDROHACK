"""
evaluation.py - Evaluation pipeline comparing Dense, BM25, and Hybrid retrieval.

This module:
1. Loads a sample of MS MARCO evaluation queries (with ground-truth relevant passages).
2. Runs each retriever on every query.
3. Computes Recall@5, Precision@5, MRR, Context Precision, Context Recall.
4. Returns a comparison table + saves results to JSON.

Note on RAGAS
-------------
RAGAS (https://docs.ragas.io/) uses LLM-based metrics for context faithfulness
and answer relevance.  For retrieval-only evaluation (no LLM required), we use
deterministic IR metrics instead, which we document as 'RAGAS-style'.
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.config import EVAL_DIR, FINAL_TOP_K
from src.metrics import evaluate_retrieval
from src.utils import get_logger, save_json

logger = get_logger(__name__)


def run_evaluation(
    dense_retriever,
    bm25_retriever,
    hybrid_retriever,
    queries: list[dict],
    k: int = FINAL_TOP_K,
    max_queries: int = 200,
    save_path: Optional[str | Path] = None,
) -> dict:
    """
    Evaluate all three retrievers on a set of queries and return metrics.

    Parameters
    ----------
    dense_retriever   : DenseRetriever instance
    bm25_retriever    : BM25Retriever instance
    hybrid_retriever  : HybridRetriever instance
    queries           : list of query dicts from data_loader.load_ms_marco_queries()
    k                 : evaluation cutoff (top-k)
    max_queries       : limit evaluation to this many queries (speed)
    save_path         : if provided, save results JSON here

    Returns
    -------
    dict with keys:
        'dense'   : metric dict
        'bm25'    : metric dict
        'hybrid'  : metric dict
        'table'   : formatted comparison table as string
    """
    queries = queries[:max_queries]
    logger.info(
        f"Evaluating {len(queries)} queries with k={k} …"
    )

    # Collect per-query results for each retriever
    dense_query_results  = []
    bm25_query_results   = []
    hybrid_query_results = []

    for i, q in enumerate(queries):
        query_text     = q["query_text"]
        relevant_texts = q["relevant_passage_texts"]

        if not query_text.strip() or not relevant_texts:
            continue

        try:
            # Dense
            dense_res = dense_retriever.search(query_text, top_k=k)
            dense_query_results.append(
                {"retrieved": dense_res, "relevant_texts": relevant_texts}
            )

            # BM25
            bm25_res = bm25_retriever.search(query_text, top_k=k)
            bm25_query_results.append(
                {"retrieved": bm25_res, "relevant_texts": relevant_texts}
            )

            # Hybrid
            hybrid_response = hybrid_retriever.search(query_text)
            hybrid_res = hybrid_response["results"]
            hybrid_query_results.append(
                {"retrieved": hybrid_res, "relevant_texts": relevant_texts}
            )

        except Exception as e:
            logger.warning(f"Query {i} failed: {e}")
            continue

        if (i + 1) % 20 == 0:
            logger.info(f"  Evaluated {i + 1}/{len(queries)} queries …")

    # Compute aggregate metrics
    dense_metrics  = evaluate_retrieval(dense_query_results,  k=k)
    bm25_metrics   = evaluate_retrieval(bm25_query_results,   k=k)
    hybrid_metrics = evaluate_retrieval(hybrid_query_results, k=k)

    # Format comparison table
    table = _format_table(dense_metrics, bm25_metrics, hybrid_metrics, k)
    logger.info("\n" + table)

    results = {
        "dense":   dense_metrics,
        "bm25":    bm25_metrics,
        "hybrid":  hybrid_metrics,
        "table":   table,
        "k":       k,
        "n_queries_evaluated": len(dense_query_results),
    }

    # Save results to disk
    out_path = Path(save_path) if save_path else Path(EVAL_DIR) / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    logger.info(f"Evaluation results saved to {out_path}")

    return results


def _format_table(
    dense: dict,
    bm25: dict,
    hybrid: dict,
    k: int,
) -> str:
    """Format a comparison table as a text string."""
    header = (
        f"{'Method':<12} | {'Recall@'+str(k):<12} | {'Precision@'+str(k):<14} | "
        f"{'MRR':<8} | {'Ctx Precision':<14} | {'Ctx Recall':<12}"
    )
    sep = "-" * len(header)

    def row(name, m):
        return (
            f"{name:<12} | {m['recall_at_k']:<12.4f} | "
            f"{m['precision_at_k']:<14.4f} | {m['mrr']:<8.4f} | "
            f"{m['context_precision']:<14.4f} | {m['context_recall']:<12.4f}"
        )

    return "\n".join(
        [
            sep,
            header,
            sep,
            row("Dense",  dense),
            row("BM25",   bm25),
            row("Hybrid", hybrid),
            sep,
        ]
    )
