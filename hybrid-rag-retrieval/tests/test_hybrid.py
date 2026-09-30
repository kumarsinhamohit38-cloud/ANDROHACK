"""
tests/test_hybrid.py - Unit tests for HybridRetriever and metadata filtering.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rrf import reciprocal_rank_fusion
from src.metadata_filter import (
    build_chroma_filter,
    apply_filter,
    renumber_ranks,
)
from src.hybrid_search import HybridRetriever


# ─────────────────────────────────────────────────────────────────────────────
# Metadata filter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildChromaFilter:

    def test_no_args_returns_none(self):
        assert build_chroma_filter() is None

    def test_single_condition(self):
        f = build_chroma_filter(source="ms_marco")
        assert f == {"source": "ms_marco"}

    def test_two_conditions_use_and(self):
        f = build_chroma_filter(source="ms_marco", is_selected=True)
        assert "$and" in f
        assert len(f["$and"]) == 2

    def test_three_conditions(self):
        f = build_chroma_filter(source="ms_marco", dataset="ms_marco", is_selected=False)
        assert "$and" in f
        assert len(f["$and"]) == 3


class TestApplyFilter:

    RESULTS = [
        {"document_id": "p0", "text": "a", "score": 1.0, "rank": 1,
         "retrieval_method": "dense",
         "metadata": {"source": "ms_marco", "is_selected": True}},
        {"document_id": "p1", "text": "b", "score": 0.9, "rank": 2,
         "retrieval_method": "dense",
         "metadata": {"source": "ms_marco", "is_selected": False}},
        {"document_id": "p2", "text": "c", "score": 0.8, "rank": 3,
         "retrieval_method": "dense",
         "metadata": {"source": "other", "is_selected": False}},
    ]

    def test_no_filter_returns_all(self):
        assert apply_filter(self.RESULTS) == self.RESULTS

    def test_filter_by_source(self):
        filtered = apply_filter(self.RESULTS, source="ms_marco")
        assert len(filtered) == 2
        assert all(r["metadata"]["source"] == "ms_marco" for r in filtered)

    def test_filter_by_is_selected(self):
        filtered = apply_filter(self.RESULTS, is_selected=True)
        assert len(filtered) == 1
        assert filtered[0]["document_id"] == "p0"

    def test_filter_removes_all(self):
        filtered = apply_filter(self.RESULTS, source="nonexistent")
        assert filtered == []

    def test_filter_preserves_order(self):
        filtered = apply_filter(self.RESULTS, source="ms_marco")
        ids = [r["document_id"] for r in filtered]
        assert ids == ["p0", "p1"]

    def test_custom_filter(self):
        filtered = apply_filter(self.RESULTS, custom={"is_selected": False})
        assert len(filtered) == 2

    def test_empty_results(self):
        assert apply_filter([], source="ms_marco") == []


class TestRenumberRanks:

    def test_renumber_after_filtering(self):
        results = [
            {"document_id": "p0", "rank": 1},
            {"document_id": "p2", "rank": 3},  # rank 2 was filtered out
        ]
        renumbered = renumber_ranks(results)
        assert renumbered[0]["rank"] == 1
        assert renumbered[1]["rank"] == 2

    def test_empty_list(self):
        assert renumber_ranks([]) == []


class TestHybridRetrieverFiltering:

    def test_filter_can_select_candidates_beyond_three_times_top_k(self):
        """Post-fusion filters should still fill Top-K from all candidates."""
        class Dense:
            def search(self, query, top_k):
                return [
                    {"document_id": f"d{i}", "rank": i, "score": 1.0 / i}
                    for i in range(1, 11)
                ]

        class BM25:
            def search(self, query, top_k):
                return []

        class Store:
            def get_by_ids(self, ids):
                return [
                    {
                        "document_id": doc_id,
                        "text": doc_id,
                        "metadata": {"keep": doc_id in {"d7", "d8", "d9", "d10"}},
                    }
                    for doc_id in ids
                ]

        retriever = HybridRetriever(
            dense_retriever=Dense(), bm25_retriever=BM25(), vector_store=Store(),
            dense_top_k=10, bm25_top_k=10, final_top_k=4,
        )

        response = retriever.search("test", metadata_filter={"keep": True})
        assert [item["document_id"] for item in response["results"]] == ["d7", "d8", "d9", "d10"]


# ─────────────────────────────────────────────────────────────────────────────
# RRF integration (without loading models/indexes)
# ─────────────────────────────────────────────────────────────────────────────

class TestRRFIntegration:
    """
    Test the full RRF pipeline logic using mocked ranked lists.
    No embedding model or vector store needed.
    """

    def test_rrf_rewards_overlap(self):
        """
        Documents appearing in both dense and BM25 lists should score higher
        than documents appearing in only one list.
        """
        dense_ranking = ["doc_a", "doc_b", "doc_c", "doc_d"]
        bm25_ranking  = ["doc_b", "doc_e", "doc_c", "doc_f"]

        fused = dict(reciprocal_rank_fusion([dense_ranking, bm25_ranking]))

        # doc_b: rank 2 in dense + rank 1 in bm25
        # doc_a: rank 1 in dense only
        # doc_e: rank 2 in bm25 only
        assert fused["doc_b"] > fused["doc_a"]
        assert fused["doc_b"] > fused["doc_e"]

    def test_rrf_top_result_is_consistent(self):
        """Top fused result should be the same regardless of list order."""
        list_a = ["winner", "second", "third"]
        list_b = ["winner", "fourth", "fifth"]

        fused_1 = reciprocal_rank_fusion([list_a, list_b])
        fused_2 = reciprocal_rank_fusion([list_b, list_a])

        assert fused_1[0][0] == "winner"
        assert fused_2[0][0] == "winner"

    def test_rrf_with_empty_bm25_ranking(self):
        """If BM25 returns no results, dense results should still be fused."""
        dense_ranking = ["doc_a", "doc_b"]
        bm25_ranking  = []

        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
        ids = [doc_id for doc_id, _ in fused]
        assert "doc_a" in ids
        assert "doc_b" in ids

    def test_rrf_with_empty_dense_ranking(self):
        """If dense returns no results, BM25 results should still be fused."""
        dense_ranking = []
        bm25_ranking  = ["doc_a", "doc_b"]

        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
        ids = [doc_id for doc_id, _ in fused]
        assert "doc_a" in ids

    def test_rrf_both_empty_returns_empty(self):
        """Two empty rankings should produce empty output."""
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []
