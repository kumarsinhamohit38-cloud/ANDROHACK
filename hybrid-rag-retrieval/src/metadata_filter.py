"""
metadata_filter.py - Metadata filtering utilities.

Metadata filtering reduces the retrieval search space to a specific subset
of passages BEFORE or AFTER retrieval, depending on the backend.

ChromaDB supports server-side pre-filtering (the 'where' clause).
BM25 applies client-side post-filtering.

Supported filter fields (from MS MARCO metadata):
    - source      : "ms_marco"
    - dataset     : "ms_marco"
    - is_selected : True / False  (relevance label from MS MARCO)

Usage
-----
    from src.metadata_filter import build_filter, apply_filter

    # Build a ChromaDB-compatible filter dict
    chroma_filter = build_filter(source="ms_marco")

    # Apply post-hoc filter to a list of result dicts
    filtered = apply_filter(results, is_selected=True)
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Build a ChromaDB 'where' clause
# ---------------------------------------------------------------------------

def build_chroma_filter(
    source: Optional[str] = None,
    dataset: Optional[str] = None,
    is_selected: Optional[bool] = None,
) -> Optional[dict]:
    """
    Build a ChromaDB-compatible metadata filter dict.

    Only include fields that are actually specified (not None).

    Parameters
    ----------
    source      : filter by metadata['source'], e.g. "ms_marco"
    dataset     : filter by metadata['dataset']
    is_selected : filter by metadata['is_selected'] (True = relevant passages)

    Returns
    -------
    A ChromaDB 'where' dict, or None if no filters are specified.

    Example
    -------
    >>> build_chroma_filter(source="ms_marco")
    {'source': 'ms_marco'}
    >>> build_chroma_filter(source="ms_marco", is_selected=True)
    {'$and': [{'source': 'ms_marco'}, {'is_selected': True}]}
    """
    conditions = []

    if source is not None:
        conditions.append({"source": source})

    if dataset is not None:
        conditions.append({"dataset": dataset})

    if is_selected is not None:
        conditions.append({"is_selected": is_selected})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    # Multiple conditions: combine with $and
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Post-hoc filter (for BM25 results or any list of result dicts)
# ---------------------------------------------------------------------------

def apply_filter(
    results: list[dict],
    source: Optional[str] = None,
    dataset: Optional[str] = None,
    is_selected: Optional[bool] = None,
    custom: Optional[dict] = None,
) -> list[dict]:
    """
    Filter a list of retrieval result dicts by their metadata fields.

    Parameters
    ----------
    results     : list of result dicts (each must have a 'metadata' key)
    source      : keep only results where metadata['source'] == source
    dataset     : keep only results where metadata['dataset'] == dataset
    is_selected : keep only results where metadata['is_selected'] == is_selected
    custom      : arbitrary dict of {field: value} constraints

    Returns
    -------
    Filtered list (order preserved, ranks NOT re-numbered).
    """
    filters = {}
    if source      is not None: filters["source"]      = source
    if dataset     is not None: filters["dataset"]     = dataset
    if is_selected is not None: filters["is_selected"] = is_selected
    if custom:
        filters.update(custom)

    if not filters:
        return results  # no filtering needed

    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if all(meta.get(k) == v for k, v in filters.items()):
            filtered.append(r)

    return filtered


def renumber_ranks(results: list[dict]) -> list[dict]:
    """
    Re-number the 'rank' field in a result list starting from 1.
    Useful after filtering removes some items.
    """
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results
