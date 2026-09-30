"""
preprocessing.py - Text preprocessing for BM25 and vector indexing.

We keep preprocessing intentionally light:
  - For BM25:  lowercase + simple whitespace tokenisation (no stemming/stopwords
               so we don't accidentally remove important domain terms).
  - For dense: keep original text – the embedding model handles normalisation.

A passage dict produced here looks like:

    {
        "id":       "passage_0",
        "text":     "Original passage text …",        # for display & dense embed
        "tokens":   ["original", "passage", "text"],  # for BM25
        "metadata": { … }                             # forwarded as-is
    }
"""

import re
from typing import Optional

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokeniser for BM25
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[^\w]+")  # split on any non-word character


def tokenize(text: str) -> list[str]:
    """
    Lightweight tokeniser for BM25.

    Steps
    -----
    1. Lowercase
    2. Split on non-word characters (punctuation, whitespace, …)
    3. Remove empty tokens and single-character tokens (noise)

    We intentionally do NOT:
    - Stem / lemmatise  (would lose exact-match signal)
    - Remove stopwords  (BM25 handles term frequency naturally)
    """
    tokens = _SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) > 1]


def clean_text(text: str) -> str:
    """
    Minimal text cleaning for the passage text used in display and embedding.

    - Strip leading/trailing whitespace
    - Collapse multiple spaces/newlines into a single space
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------

def preprocess_passages(passages: list[dict]) -> list[dict]:
    """
    Given a list of raw passage dicts (from data_loader), return a list of
    preprocessed passage dicts ready for indexing.

    Each output dict has:
        id       : str   – unique passage ID
        text     : str   – cleaned original text (for display & dense embed)
        tokens   : list  – tokenised text for BM25
        metadata : dict  – original metadata plus 'text_length' field
    """
    logger.info(f"Preprocessing {len(passages)} passages …")
    processed = []

    for passage in passages:
        raw_text = passage.get("text", "")
        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        tokens  = tokenize(cleaned)

        # Enrich metadata
        meta = dict(passage.get("metadata", {}))
        meta["text_length"] = len(cleaned)
        meta["token_count"] = len(tokens)

        processed.append(
            {
                "id":       passage["id"],
                "text":     cleaned,
                "tokens":   tokens,
                "metadata": meta,
            }
        )

    logger.info(f"Preprocessing done. {len(processed)} passages ready.")
    return processed


def get_texts(passages: list[dict]) -> list[str]:
    """Extract the text field from each passage (for batch embedding)."""
    return [p["text"] for p in passages]


def get_token_lists(passages: list[dict]) -> list[list[str]]:
    """Extract the tokens field from each passage (for BM25 indexing)."""
    return [p["tokens"] for p in passages]


def get_ids(passages: list[dict]) -> list[str]:
    """Extract passage IDs."""
    return [p["id"] for p in passages]
