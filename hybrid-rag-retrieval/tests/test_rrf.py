"""
tests/test_rrf.py - Unit tests for Reciprocal Rank Fusion.

Tests cover:
- Single ranking list
- Two ranking lists
- Documents appearing in both lists (should get higher score)
- Score ordering
- Custom k values
- Empty inputs
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rrf import reciprocal_rank_fusion, rrf_with_details


# ─────────────────────────────────────────────────────────────────────────────
# Basic functionality
# ─────────────────────────────────────────────────────────────────────────────

class TestRRFBasic:

    def test_single_ranking_returns_all_docs(self):
        """With one list, all docs should appear in output."""
        ranking = ["doc_a", "doc_b", "doc_c"]
        results = reciprocal_rank_fusion([ranking])
        result_ids = [doc_id for doc_id, _ in results]
        assert result_ids == ranking  # same order as input (rank 1 wins)

    def test_single_ranking_scores_decreasing(self):
        """Scores should decrease as rank increases."""
        ranking = ["doc_a", "doc_b", "doc_c"]
        results = reciprocal_rank_fusion([ranking])
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_single_ranking_correct_score(self):
        """Verify the exact score formula: 1/(k+rank)."""
        k = 60
        ranking = ["doc_a", "doc_b"]
        results = reciprocal_rank_fusion([ranking], k=k)
        score_map = dict(results)

        expected_a = 1.0 / (k + 1)  # rank 1
        expected_b = 1.0 / (k + 2)  # rank 2

        assert abs(score_map["doc_a"] - expected_a) < 1e-10
        assert abs(score_map["doc_b"] - expected_b) < 1e-10

    def test_two_rankings_document_in_both_gets_higher_score(self):
        """
        A document appearing in both lists should have a higher score than
        a document appearing in only one list.

        ranking_1 = ["A", "B", "C"]
        ranking_2 = ["B", "C", "D"]

        → B appears in both (rank 2 and rank 1)  → higher score
        → A appears only in ranking_1
        → D appears only in ranking_2
        """
        ranking_1 = ["A", "B", "C"]
        ranking_2 = ["B", "C", "D"]
        results = reciprocal_rank_fusion([ranking_1, ranking_2])
        score_map = dict(results)

        # B appears in both lists → should outrank A (only in list 1)
        assert score_map["B"] > score_map["A"]
        # B appears in both lists → should outrank D (only in list 2)
        assert score_map["B"] > score_map["D"]

    def test_output_is_sorted_descending(self):
        """Output must always be sorted by descending score."""
        ranking_1 = ["A", "B", "C", "D"]
        ranking_2 = ["C", "D", "E", "A"]
        results = reciprocal_rank_fusion([ranking_1, ranking_2])
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input_returns_empty(self):
        """Empty list of rankings → empty output."""
        results = reciprocal_rank_fusion([])
        assert results == []

    def test_single_document_single_list(self):
        """One document in one list → that document returned."""
        results = reciprocal_rank_fusion([["only_doc"]])
        assert len(results) == 1
        assert results[0][0] == "only_doc"

    def test_all_unique_documents(self):
        """
        If no document appears in more than one list, scores should be
        proportional to rank only (no cross-list bonus).
        """
        ranking_1 = ["A", "B"]
        ranking_2 = ["C", "D"]
        results = reciprocal_rank_fusion([ranking_1, ranking_2])
        score_map = dict(results)

        k = 60
        assert abs(score_map["A"] - 1 / (k + 1)) < 1e-10
        assert abs(score_map["C"] - 1 / (k + 1)) < 1e-10

    def test_custom_k_parameter(self):
        """
        Lower k → bigger differences between ranks.
        Higher k → smaller differences between ranks.
        """
        ranking = ["A", "B"]

        results_k1  = dict(reciprocal_rank_fusion([ranking], k=1))
        results_k60 = dict(reciprocal_rank_fusion([ranking], k=60))

        # With k=1:  score_A = 1/2, score_B = 1/3.  Diff = 1/6
        diff_k1  = results_k1["A"]  - results_k1["B"]
        # With k=60: score_A = 1/61, score_B = 1/62.  Diff ≈ 0.00026
        diff_k60 = results_k60["A"] - results_k60["B"]

        assert diff_k1 > diff_k60

    def test_k_default_is_60(self):
        """Default k should be 60 (as in the original RRF paper)."""
        ranking = ["X"]
        results = dict(reciprocal_rank_fusion([ranking]))
        expected = 1.0 / (60 + 1)
        assert abs(results["X"] - expected) < 1e-10

    def test_three_lists_document_in_all_gets_max_score(self):
        """A document in all three lists should have the highest score."""
        ranking_1 = ["winner", "loser1"]
        ranking_2 = ["winner", "loser2"]
        ranking_3 = ["winner", "loser3"]
        results = dict(reciprocal_rank_fusion([ranking_1, ranking_2, ranking_3]))

        assert results["winner"] > results["loser1"]
        assert results["winner"] > results["loser2"]
        assert results["winner"] > results["loser3"]

    def test_duplicate_ids_in_same_ranking(self):
        """
        If a doc appears twice in the same ranking (shouldn't happen in
        practice), both positions are counted.
        """
        ranking = ["A", "A", "B"]
        results = dict(reciprocal_rank_fusion([ranking]))
        # A at rank 1 and rank 2: 1/61 + 1/62
        expected_a = 1 / (60 + 1) + 1 / (60 + 2)
        assert abs(results["A"] - expected_a) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# rrf_with_details
# ─────────────────────────────────────────────────────────────────────────────

class TestRRFWithDetails:

    def test_tied_scores_have_stable_first_seen_order(self):
        """Tied documents must not depend on Python set iteration order."""
        results = rrf_with_details([["A"], ["B"]])
        assert [result["document_id"] for result in results] == ["A", "B"]

    def test_returns_per_list_ranks(self):
        """rrf_with_details should include 'ranks' showing per-list position."""
        ranking_1 = ["A", "B"]
        ranking_2 = ["B", "A"]
        results = rrf_with_details([ranking_1, ranking_2])
        result_map = {r["document_id"]: r for r in results}

        # A is rank 1 in list 0, rank 2 in list 1
        assert result_map["A"]["ranks"][0] == 1
        assert result_map["A"]["ranks"][1] == 2

        # B is rank 2 in list 0, rank 1 in list 1
        assert result_map["B"]["ranks"][0] == 2
        assert result_map["B"]["ranks"][1] == 1

    def test_none_rank_for_missing_doc(self):
        """A doc in only one list should have None rank for the other list."""
        ranking_1 = ["A", "B"]
        ranking_2 = ["C"]
        results = rrf_with_details([ranking_1, ranking_2])
        result_map = {r["document_id"]: r for r in results}

        # A not in ranking_2 → rank for list 1 is None
        assert result_map["A"]["ranks"][1] is None
        # C not in ranking_1 → rank for list 0 is None
        assert result_map["C"]["ranks"][0] is None

    def test_scores_match_basic_rrf(self):
        """rrf_with_details scores should match reciprocal_rank_fusion."""
        ranking_1 = ["A", "B", "C"]
        ranking_2 = ["B", "D"]

        basic   = dict(reciprocal_rank_fusion([ranking_1, ranking_2]))
        details = {r["document_id"]: r["rrf_score"] for r in rrf_with_details([ranking_1, ranking_2])}

        for doc_id, score in basic.items():
            assert abs(details[doc_id] - score) < 1e-10
