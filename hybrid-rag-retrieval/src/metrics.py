"""
metrics.py - Retrieval quality metrics computed without an LLM.

We implement the following standard IR metrics deterministically:

  Recall@K    - What fraction of relevant passages were retrieved in top-K?
  Precision@K - Of the K retrieved passages, what fraction are relevant?
  MRR         - Mean Reciprocal Rank: average of 1/rank of first relevant hit.

These metrics are computed by checking whether retrieved passage texts
overlap with the known-relevant passage texts from MS MARCO annotations.

RAGAS-style naming
------------------
In RAGAS terminology:
  "Context Precision" ≈ Precision@K  (are retrieved passages relevant?)
  "Context Recall"    ≈ Recall@K     (were all relevant passages retrieved?)

Because RAGAS requires an LLM for its full implementation, we compute
these metrics deterministically using exact-text matching + substring
matching, which is appropriate for MS MARCO where we know the ground-truth
passage texts.

All functions operate on lists of result dicts (from our retrieval pipeline)
and lists of relevant-passage-text strings (from MS MARCO annotations).
"""

from typing import Optional


def _is_relevant(retrieved_text: str, relevant_texts: list[str]) -> bool:
    """
    Check if a retrieved passage is relevant.

    Strategy (in order):
    1. Exact text match (after strip)
    2. Substring match in either direction (relevant ⊆ retrieved or vice versa)

    This is intentionally lenient to handle minor text normalisation differences.
    """
    retrieved_text = retrieved_text.strip().lower()
    for rel_text in relevant_texts:
        rel_lower = rel_text.strip().lower()
        if retrieved_text == rel_lower:
            return True
        if rel_lower in retrieved_text or retrieved_text in rel_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved: list[dict],
    relevant_texts: list[str],
    k: int,
) -> float:
    """
    Recall@K: fraction of relevant passages found in the top-K results.

    Recall@K = |{retrieved top-K} ∩ relevant| / |relevant|

    Parameters
    ----------
    retrieved      : list of result dicts (ordered by rank)
    relevant_texts : list of ground-truth relevant passage texts
    k              : cutoff rank

    Returns
    -------
    float in [0, 1].  Returns 0.0 if relevant_texts is empty.
    """
    if not relevant_texts:
        return 0.0

    top_k = retrieved[:k]
    n_relevant_found = sum(
        1 for r in top_k if _is_relevant(r["text"], relevant_texts)
    )
    return n_relevant_found / len(relevant_texts)


def precision_at_k(
    retrieved: list[dict],
    relevant_texts: list[str],
    k: int,
) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.

    Precision@K = |{retrieved top-K} ∩ relevant| / K

    Returns float in [0, 1].  Returns 0.0 if retrieved is empty.
    """
    if not retrieved:
        return 0.0

    top_k = retrieved[:k]
    n_relevant_found = sum(
        1 for r in top_k if _is_relevant(r["text"], relevant_texts)
    )
    return n_relevant_found / min(k, len(retrieved))


def reciprocal_rank(
    retrieved: list[dict],
    relevant_texts: list[str],
) -> float:
    """
    Reciprocal Rank for a single query.

    RR = 1 / rank_of_first_relevant_result
       = 0 if no relevant result found

    Returns float in [0, 1].
    """
    for i, r in enumerate(retrieved, start=1):
        if _is_relevant(r["text"], relevant_texts):
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# RAGAS-style metrics (deterministic)
# ---------------------------------------------------------------------------

def context_precision_at_k(
    retrieved: list[dict],
    relevant_texts: list[str],
    k: int = 5,
) -> float:
    """
    RAGAS-style Context Precision (deterministic version).

    Measures: of the contexts retrieved, how many are actually relevant?
    Equivalent to Precision@K here.
    """
    return precision_at_k(retrieved, relevant_texts, k)


def context_recall_at_k(
    retrieved: list[dict],
    relevant_texts: list[str],
    k: int = 5,
) -> float:
    """
    RAGAS-style Context Recall (deterministic version).

    Measures: were all relevant contexts retrieved within top-K?
    Equivalent to Recall@K here.
    """
    return recall_at_k(retrieved, relevant_texts, k)


# ---------------------------------------------------------------------------
# Aggregate over multiple queries
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    query_results: list[dict],
    k: int = 5,
) -> dict:
    """
    Compute mean retrieval metrics over a list of query evaluation results.

    Parameters
    ----------
    query_results : list of dicts, each with:
        'retrieved'       : list of result dicts (from retrieval pipeline)
        'relevant_texts'  : list of ground-truth relevant passage texts

    k             : evaluation cutoff

    Returns
    -------
    dict with keys:
        recall_at_k, precision_at_k, mrr,
        context_precision, context_recall,
        n_queries
    """
    if not query_results:
        return {
            "recall_at_k":        0.0,
            "precision_at_k":     0.0,
            "mrr":                0.0,
            "context_precision":  0.0,
            "context_recall":     0.0,
            "n_queries":          0,
        }

    recalls    = []
    precisions = []
    rrs        = []

    for qr in query_results:
        retrieved      = qr["retrieved"]
        relevant_texts = qr["relevant_texts"]

        recalls.append(recall_at_k(retrieved, relevant_texts, k))
        precisions.append(precision_at_k(retrieved, relevant_texts, k))
        rrs.append(reciprocal_rank(retrieved, relevant_texts))

    n = len(query_results)
    mean_recall    = sum(recalls)    / n
    mean_precision = sum(precisions) / n
    mean_rr        = sum(rrs)        / n

    return {
        "recall_at_k":       round(mean_recall, 4),
        "precision_at_k":    round(mean_precision, 4),
        "mrr":               round(mean_rr, 4),
        "context_precision": round(mean_precision, 4),   # alias
        "context_recall":    round(mean_recall, 4),      # alias
        "n_queries":         n,
    }
