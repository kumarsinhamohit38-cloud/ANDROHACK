"""
tests/test_bm25.py - Unit tests for BM25Retriever.

Tests cover:
- Building the index
- Searching with a simple query
- Top-k cutoff
- Empty query
- Score ordering
- Metadata filtering
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bm25_search import BM25Retriever


# ─────────────────────────────────────────────────────────────────────────────
# Sample dataset for testing
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_PASSAGES = [
    {
        "id": "p0",
        "text": "Paris is the capital city of France.",
        "tokens": ["paris", "is", "the", "capital", "city", "of", "france"],
        "metadata": {"source": "ms_marco", "dataset": "ms_marco", "is_selected": True},
    },
    {
        "id": "p1",
        "text": "Berlin is the capital city of Germany.",
        "tokens": ["berlin", "is", "the", "capital", "city", "of", "germany"],
        "metadata": {"source": "ms_marco", "dataset": "ms_marco", "is_selected": False},
    },
    {
        "id": "p2",
        "text": "London is the capital city of the United Kingdom.",
        "tokens": ["london", "is", "the", "capital", "city", "of", "the", "united", "kingdom"],
        "metadata": {"source": "ms_marco", "dataset": "ms_marco", "is_selected": False},
    },
    {
        "id": "p3",
        "text": "Tokyo is the capital city of Japan.",
        "tokens": ["tokyo", "is", "the", "capital", "city", "of", "japan"],
        "metadata": {"source": "other", "dataset": "other", "is_selected": False},
    },
    {
        "id": "p4",
        "text": "Machine learning models can hallucinate facts.",
        "tokens": ["machine", "learning", "models", "can", "hallucinate", "facts"],
        "metadata": {"source": "ms_marco", "dataset": "ms_marco", "is_selected": False},
    },
]


@pytest.fixture(scope="module")
def bm25():
    """Build and return a BM25Retriever over the sample passages."""
    retriever = BM25Retriever()
    retriever.build(SAMPLE_PASSAGES)
    return retriever


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBM25Build:

    def test_build_marks_ready(self):
        retriever = BM25Retriever()
        assert not retriever.is_ready()
        retriever.build(SAMPLE_PASSAGES)
        assert retriever.is_ready()

    def test_correct_doc_count(self):
        retriever = BM25Retriever()
        retriever.build(SAMPLE_PASSAGES)
        assert len(retriever._doc_ids) == len(SAMPLE_PASSAGES)


class TestBM25Search:

    def test_capital_france_returns_paris_first(self, bm25):
        """Query 'capital France' should rank the Paris passage first."""
        results = bm25.search("capital France", top_k=5)
        assert results[0]["document_id"] == "p0"

    def test_returns_correct_number_of_results(self, bm25):
        """top_k should limit the number of results."""
        results = bm25.search("capital city", top_k=3)
        assert len(results) <= 3

    def test_results_have_required_fields(self, bm25):
        """Each result must have the standard fields."""
        results = bm25.search("capital", top_k=2)
        for r in results:
            assert "document_id" in r
            assert "text" in r
            assert "score" in r
            assert "rank" in r
            assert "retrieval_method" in r
            assert r["retrieval_method"] == "bm25"

    def test_scores_are_descending(self, bm25):
        """Scores must be in descending order (best match first)."""
        results = bm25.search("capital city", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self, bm25):
        """Ranks should start at 1 and increment by 1."""
        results = bm25.search("capital", top_k=5)
        for i, r in enumerate(results, start=1):
            assert r["rank"] == i

    def test_empty_query_returns_empty(self, bm25):
        """An empty string query should return an empty list."""
        results = bm25.search("", top_k=5)
        assert results == []

    def test_whitespace_query_returns_empty(self, bm25):
        """A whitespace-only query should return an empty list."""
        results = bm25.search("   ", top_k=5)
        assert results == []

    def test_unrelated_query_returns_low_scores(self, bm25):
        """A completely unrelated query should return lower scores."""
        results_related   = bm25.search("capital France", top_k=1)
        results_unrelated = bm25.search("xyzzy foobar", top_k=1)

        if results_related and results_unrelated:
            assert results_related[0]["score"] >= results_unrelated[0]["score"]

    def test_metadata_filter_applied(self, bm25):
        """
        When filtering by source='ms_marco', the passage with source='other'
        (p3 = Tokyo) should not appear in results.
        """
        results = bm25.search("capital city", top_k=5, metadata_filter={"source": "ms_marco"})
        result_ids = [r["document_id"] for r in results]
        assert "p3" not in result_ids

    def test_get_ranked_ids_returns_ids_only(self, bm25):
        """get_ranked_ids should return a list of string IDs."""
        ids = bm25.get_ranked_ids("capital city", top_k=3)
        assert isinstance(ids, list)
        assert all(isinstance(i, str) for i in ids)


class TestBM25Persistence:

    def test_save_and_load(self, tmp_path, bm25):
        """Saving and loading the index should preserve search results."""
        index_path = tmp_path / "test_bm25.pkl"
        bm25.save(index_path)

        loaded = BM25Retriever()
        loaded.load(index_path)

        original_results = bm25.search("capital France", top_k=3)
        loaded_results   = loaded.search("capital France", top_k=3)

        assert len(original_results) == len(loaded_results)
        for orig, load in zip(original_results, loaded_results):
            assert orig["document_id"] == load["document_id"]
            assert abs(orig["score"] - load["score"]) < 1e-8

    def test_load_missing_file_raises(self, tmp_path):
        """Loading a non-existent file should raise FileNotFoundError."""
        retriever = BM25Retriever(index_path=tmp_path / "nonexistent.pkl")
        with pytest.raises(FileNotFoundError):
            retriever.load()
