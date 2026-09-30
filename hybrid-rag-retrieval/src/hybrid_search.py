"""
hybrid_search.py - Hybrid Dense + BM25 retrieval using Reciprocal Rank Fusion.

Architecture
------------

    query
      │
      ├─── Dense Retriever  ──► top-20 doc IDs (dense_ranking)
      │
      └─── BM25 Retriever   ──► top-20 doc IDs (bm25_ranking)
                │
                ▼
          RRF Fusion
                │
                ▼
      Fetch full passage info
                │
                ▼
       Metadata Filtering (optional)
                │
                ▼
          Top-5 results

This is the core pipeline.  The DenseRetriever and BM25Retriever run in
parallel (they are independent), and their ranked lists are fused with RRF.
"""

import time
from typing import Optional

from src.config import BM25_TOP_K, DENSE_TOP_K, FINAL_TOP_K, RRF_K
from src.bm25_search import BM25Retriever
from src.dense_search import DenseRetriever
from src.rrf import rrf_with_details
from src.vector_store import VectorStore
from src.metadata_filter import apply_filter, renumber_ranks
from src.utils import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Combines dense vector search + BM25 keyword search using RRF.

    Parameters
    ----------
    dense_retriever : DenseRetriever instance
    bm25_retriever  : BM25Retriever instance
    vector_store    : VectorStore instance (for fetching passage text by ID)
    dense_top_k     : candidates from dense search
    bm25_top_k      : candidates from BM25 search
    final_top_k     : number of final results to return
    rrf_k           : RRF smoothing constant
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_store: Optional[VectorStore] = None,
        dense_top_k: int = DENSE_TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        final_top_k: int = FINAL_TOP_K,
        rrf_k: int = RRF_K,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever  = bm25_retriever  or BM25Retriever()
        self.vector_store    = vector_store    or VectorStore()

        self.dense_top_k = dense_top_k
        self.bm25_top_k  = bm25_top_k
        self.final_top_k = final_top_k
        self.rrf_k       = rrf_k

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
    ) -> dict:
        """
        Run hybrid retrieval and return results with timing breakdown.

        Parameters
        ----------
        query           : str – user's query
        metadata_filter : optional dict for metadata filtering

        Returns
        -------
        dict with keys:
            results          : list of result dicts (top-k)
            latency          : dict of timing info (seconds)
        """
        if not query.strip():
            return {"results": [], "latency": {}}

        timings = {}

        # ── Step 1: Dense retrieval ────────────────────────────────────
        t0 = time.perf_counter()
        dense_results = self.dense_retriever.search(
            query, top_k=self.dense_top_k
        )
        timings["dense_ms"] = (time.perf_counter() - t0) * 1000
        dense_ids = [r["document_id"] for r in dense_results]
        dense_rank_lookup = {r["document_id"]: r["rank"] for r in dense_results}
        dense_score_lookup = {r["document_id"]: r["score"] for r in dense_results}

        # ── Step 2: BM25 retrieval ─────────────────────────────────────
        t0 = time.perf_counter()
        bm25_results = self.bm25_retriever.search(
            query, top_k=self.bm25_top_k
        )
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000
        bm25_ids = [r["document_id"] for r in bm25_results]
        bm25_rank_lookup  = {r["document_id"]: r["rank"]  for r in bm25_results}
        bm25_score_lookup = {r["document_id"]: r["score"] for r in bm25_results}

        # ── Step 3: RRF fusion ─────────────────────────────────────────
        t0 = time.perf_counter()
        fused = rrf_with_details(
            rankings=[dense_ids, bm25_ids],
            k=self.rrf_k,
        )
        timings["rrf_ms"] = (time.perf_counter() - t0) * 1000

        # ── Step 4: Fetch full passage information ─────────────────────
        t0 = time.perf_counter()
        # Hydrate every fused candidate before applying post-fusion metadata
        # filtering.  Restricting this to (top_k * 3) can return fewer than
        # top_k results even when matching documents exist later in the
        # dense/BM25 candidate union.
        top_fused = fused
        top_ids   = [item["document_id"] for item in top_fused]

        # Build a lookup from document_id → fused RRF info
        fused_lookup = {item["document_id"]: item for item in top_fused}

        # Fetch texts and metadata from ChromaDB
        fetched = self.vector_store.get_by_ids(top_ids)
        fetched_lookup = {f["document_id"]: f for f in fetched}
        timings["fetch_ms"] = (time.perf_counter() - t0) * 1000

        # ── Step 5: Assemble results ───────────────────────────────────
        t0 = time.perf_counter()
        results = []
        for doc_id in top_ids:
            fused_info  = fused_lookup.get(doc_id, {})
            fetched_doc = fetched_lookup.get(doc_id, {})

            result = {
                "document_id":      doc_id,
                "text":             fetched_doc.get("text", ""),
                "score":            fused_info.get("rrf_score", 0.0),
                "rank":             0,          # assigned below
                "retrieval_method": "hybrid",
                "metadata":         fetched_doc.get("metadata", {}),
                # Explainability fields
                "rrf_score":   fused_info.get("rrf_score", 0.0),
                "dense_rank":  dense_rank_lookup.get(doc_id),
                "bm25_rank":   bm25_rank_lookup.get(doc_id),
                "dense_score": dense_score_lookup.get(doc_id),
                "bm25_score":  bm25_score_lookup.get(doc_id),
            }
            results.append(result)

        # ── Step 6: Metadata filtering ─────────────────────────────────
        if metadata_filter:
            results = apply_filter(results, custom=metadata_filter)

        # Trim to final_top_k and assign ranks
        results = results[: self.final_top_k]
        results = renumber_ranks(results)
        timings["filter_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = sum(timings.values())

        return {"results": results, "latency": timings}
