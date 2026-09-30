"""
rrf.py - Reciprocal Rank Fusion (RRF).

RRF is a simple, parameter-light method for combining multiple ranked lists.
It was introduced by Cormack, Clarke & Buettcher (2009).

The key formula is:

    RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                   over all ranked lists i

where rank_i(doc) is the position of doc in list i (1-indexed),
and k is a smoothing constant (default 60).

Why RRF?
--------
- Does NOT require score normalisation between lists (scores from
  dense search and BM25 are on very different scales).
- Robust: a document that ranks well in both lists gets a high fused score.
- Simple: no learning required, no tuning beyond k.
"""


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings : list of ranked document-ID lists.
               Each inner list is ordered from most to least relevant.
               A document may appear in none, one, or all lists.
    k        : int, smoothing constant (default 60, as in the original paper).
               Higher k → less influence of high-ranked documents.

    Returns
    -------
    Sorted list of (document_id, rrf_score) tuples, highest score first.

    Examples
    --------
    >>> ranking_a = ["doc1", "doc2", "doc3"]
    >>> ranking_b = ["doc2", "doc3", "doc4"]
    >>> results = reciprocal_rank_fusion([ranking_a, ranking_b])
    >>> [doc_id for doc_id, _ in results[:2]]
    ['doc2', 'doc3']   # doc2 and doc3 appear in both lists → rewarded
    """
    # Accumulate RRF scores across all lists
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            # 1-indexed rank: position 1 → 1/(k+1), position 2 → 1/(k+2) …
            if document_id not in scores:
                scores[document_id] = 0.0
            scores[document_id] += 1.0 / (k + rank)

    # Sort by descending RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_with_details(
    rankings: list[list[str]],
    k: int = 60,
) -> list[dict]:
    """
    Like reciprocal_rank_fusion but also returns the per-list rank of
    each document (for explainability in the UI).

    Parameters
    ----------
    rankings : list of ranked document-ID lists.
    k        : RRF smoothing constant.

    Returns
    -------
    List of dicts, sorted by descending rrf_score:
        document_id : str
        rrf_score   : float
        ranks       : list[int | None]
                      rank of this document in each input list, or None
    """
    n_lists = len(rankings)

    # Build a per-list lookup: doc_id → rank (1-indexed)
    rank_lookups: list[dict[str, int]] = [
        {doc_id: rank for rank, doc_id in enumerate(ranking, start=1)}
        for ranking in rankings
    ]

    # Collect IDs in first-seen rank order.  Using a set here makes ties
    # nondeterministic (its iteration order is intentionally unspecified),
    # which can make identical queries produce differently ordered results.
    all_docs: list[str] = []
    seen: set[str] = set()
    for ranking in rankings:
        for doc_id in ranking:
            if doc_id not in seen:
                seen.add(doc_id)
                all_docs.append(doc_id)

    results = []
    for doc_id in all_docs:
        rrf_score = 0.0
        per_list_ranks = []

        for lookup in rank_lookups:
            rank = lookup.get(doc_id)  # None if not in this list
            per_list_ranks.append(rank)
            if rank is not None:
                rrf_score += 1.0 / (k + rank)

        results.append(
            {
                "document_id": doc_id,
                "rrf_score":   rrf_score,
                "ranks":       per_list_ranks,  # [dense_rank, bm25_rank]
            }
        )

    # Sort by descending RRF score
    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results
