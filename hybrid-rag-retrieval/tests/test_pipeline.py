"""
tests/test_pipeline.py - End-to-end pipeline tests (no models/indexes needed).

Tests cover:
- Preprocessing pipeline
- Metrics computation
- Result format validation
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import tokenize, clean_text, preprocess_passages
from src.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    context_precision_at_k,
    context_recall_at_k,
    evaluate_retrieval,
)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenize:

    def test_basic_tokenization(self):
        tokens = tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase(self):
        tokens = tokenize("CAPITAL LETTERS")
        assert all(t == t.lower() for t in tokens)

    def test_removes_punctuation(self):
        tokens = tokenize("Hello, world! How are you?")
        for t in tokens:
            assert "," not in t
            assert "!" not in t

    def test_removes_single_chars(self):
        tokens = tokenize("a b c hello")
        # single-char tokens should be removed
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_numbers_kept(self):
        tokens = tokenize("temperature is 37 degrees")
        assert "37" in tokens

    def test_hyphenated_terms(self):
        # Hyphen splits the word
        tokens = tokenize("state-of-the-art retrieval")
        assert "retrieval" in tokens


class TestCleanText:

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert clean_text("hello\n\nworld") == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestPreprocessPassages:

    PASSAGES = [
        {
            "id": "p0",
            "text": "Paris is the capital of France.",
            "metadata": {"source": "ms_marco"},
        },
        {
            "id": "p1",
            "text": "  Berlin is the capital of Germany.  ",
            "metadata": {"source": "ms_marco"},
        },
        {
            "id": "p2",
            "text": "",   # empty text – should be skipped
            "metadata": {},
        },
    ]

    def test_skips_empty_texts(self):
        processed = preprocess_passages(self.PASSAGES)
        assert all(p["text"] for p in processed)
        assert len(processed) == 2  # p2 is empty, should be dropped

    def test_output_has_tokens(self):
        processed = preprocess_passages(self.PASSAGES)
        for p in processed:
            assert "tokens" in p
            assert isinstance(p["tokens"], list)

    def test_output_has_metadata_enrichment(self):
        processed = preprocess_passages(self.PASSAGES)
        for p in processed:
            assert "text_length" in p["metadata"]
            assert "token_count" in p["metadata"]

    def test_text_is_cleaned(self):
        processed = preprocess_passages(self.PASSAGES)
        # Berlin passage had leading/trailing spaces – should be stripped
        berlin = next(p for p in processed if p["id"] == "p1")
        assert berlin["text"] == "Berlin is the capital of Germany."


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────────────────────────────────────

def make_result(text: str, rank: int) -> dict:
    return {"document_id": f"doc_{rank}", "text": text, "score": 1.0 / rank, "rank": rank}


class TestRecallAtK:

    def test_perfect_recall(self):
        retrieved = [make_result("relevant text", 1)]
        assert recall_at_k(retrieved, ["relevant text"], k=5) == 1.0

    def test_zero_recall(self):
        retrieved = [make_result("unrelated text", 1)]
        assert recall_at_k(retrieved, ["relevant text"], k=5) == 0.0

    def test_partial_recall(self):
        retrieved = [
            make_result("relevant text", 1),
            make_result("unrelated", 2),
        ]
        # 1 of 2 relevant found
        assert recall_at_k(retrieved, ["relevant text", "other relevant"], k=5) == 0.5

    def test_cutoff_enforced(self):
        # relevant text is at rank 3, but k=2
        retrieved = [
            make_result("unrelated1", 1),
            make_result("unrelated2", 2),
            make_result("relevant text", 3),
        ]
        assert recall_at_k(retrieved, ["relevant text"], k=2) == 0.0

    def test_empty_relevant(self):
        retrieved = [make_result("text", 1)]
        assert recall_at_k(retrieved, [], k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["relevant"], k=5) == 0.0


class TestPrecisionAtK:

    def test_perfect_precision(self):
        retrieved = [make_result("relevant", 1)]
        assert precision_at_k(retrieved, ["relevant"], k=1) == 1.0

    def test_zero_precision(self):
        retrieved = [make_result("unrelated", 1)]
        assert precision_at_k(retrieved, ["relevant"], k=1) == 0.0

    def test_half_precision(self):
        retrieved = [
            make_result("relevant", 1),
            make_result("unrelated", 2),
        ]
        assert precision_at_k(retrieved, ["relevant"], k=2) == 0.5

    def test_empty_retrieved(self):
        assert precision_at_k([], ["relevant"], k=5) == 0.0


class TestReciprocalRank:

    def test_first_result_relevant(self):
        retrieved = [make_result("relevant", 1), make_result("other", 2)]
        assert reciprocal_rank(retrieved, ["relevant"]) == 1.0

    def test_second_result_relevant(self):
        retrieved = [make_result("other", 1), make_result("relevant", 2)]
        assert abs(reciprocal_rank(retrieved, ["relevant"]) - 0.5) < 1e-10

    def test_no_relevant(self):
        retrieved = [make_result("other", 1)]
        assert reciprocal_rank(retrieved, ["relevant"]) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], ["relevant"]) == 0.0


class TestEvaluateRetrieval:

    def test_empty_query_results(self):
        result = evaluate_retrieval([])
        assert result["n_queries"] == 0
        assert result["recall_at_k"] == 0.0

    def test_correct_mean_computation(self):
        query_results = [
            {"retrieved": [make_result("rel1", 1)], "relevant_texts": ["rel1"]},
            {"retrieved": [make_result("other", 1)], "relevant_texts": ["rel2"]},
        ]
        metrics = evaluate_retrieval(query_results, k=5)
        # First query: recall=1.0, second: recall=0.0 → mean=0.5
        assert abs(metrics["recall_at_k"] - 0.5) < 1e-6
        assert metrics["n_queries"] == 2
