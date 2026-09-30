"""
bm25_search.py - BM25 sparse keyword retrieval.

BM25 (Best Match 25) is a probabilistic ranking function that ranks
passages based on exact keyword frequency and document length normalisation.

Why BM25?
---------
Dense vector search is great at semantic similarity but can miss exact
keywords, names, IDs, or rare technical terms.  BM25 fills this gap.

Implementation uses the rank_bm25 library (BM25Okapi).
"""

import pickle
from pathlib import Path
from typing import Optional

from src.config import BM25_INDEX_PATH, BM25_TOP_K
from src.preprocessing import tokenize
from src.utils import get_logger

logger = get_logger(__name__)


class BM25Retriever:
    """
    BM25 retriever backed by rank_bm25's BM25Okapi.

    Usage
    -----
        retriever = BM25Retriever()
        retriever.build(passages)          # build index once
        retriever.save()                   # persist to disk
        # later:
        retriever.load()
        results = retriever.search("what causes fever", top_k=20)
    """

    def __init__(self, index_path: str | Path = BM25_INDEX_PATH):
        self.index_path = Path(index_path)
        self._bm25 = None          # BM25Okapi object
        self._doc_ids: list[str] = []   # maps BM25 index → passage ID
        self._doc_texts: list[str] = [] # maps BM25 index → original text
        self._doc_metas: list[dict] = []  # maps BM25 index → metadata

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(self, passages: list[dict]) -> None:
        """
        Build a BM25 index from preprocessed passage dicts.

        Parameters
        ----------
        passages : list of dicts with keys: id, text, tokens, metadata
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required. Install: pip install rank-bm25"
            )

        logger.info(f"Building BM25 index over {len(passages)} passages …")

        self._doc_ids   = [p["id"] for p in passages]
        self._doc_texts = [p["text"] for p in passages]
        self._doc_metas = [p.get("metadata", {}) for p in passages]

        # Token lists for BM25
        token_lists = [p.get("tokens", tokenize(p["text"])) for p in passages]

        self._bm25 = BM25Okapi(token_lists)
        logger.info("BM25 index built.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> None:
        """Pickle the BM25 index and associated data to disk."""
        path = Path(path) if path else self.index_path
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "bm25":      self._bm25,
            "doc_ids":   self._doc_ids,
            "doc_texts": self._doc_texts,
            "doc_metas": self._doc_metas,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: Optional[str | Path] = None) -> None:
        """Load a previously saved BM25 index from disk."""
        path = Path(path) if path else self.index_path

        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. "
                "Please run: python scripts/build_indexes.py"
            )

        logger.info(f"Loading BM25 index from {path} …")
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self._bm25      = payload["bm25"]
        self._doc_ids   = payload["doc_ids"]
        self._doc_texts = payload["doc_texts"]
        self._doc_metas = payload["doc_metas"]
        logger.info(
            f"BM25 index loaded. "
            f"{len(self._doc_ids)} passages indexed."
        )

    def is_ready(self) -> bool:
        """Return True if the BM25 index is built/loaded."""
        return self._bm25 is not None and len(self._doc_ids) > 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve the top-k passages most relevant to the query.

        Parameters
        ----------
        query           : str – the user's search query
        top_k           : int – number of results to return
        metadata_filter : optional dict of metadata constraints,
                          e.g. {"source": "ms_marco"}

        Returns
        -------
        List of result dicts:
            document_id, text, score, rank, retrieval_method, metadata
        """
        if not self.is_ready():
            raise RuntimeError(
                "BM25 index is not loaded.  "
                "Call .build(passages) or .load() first."
            )

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning("Empty query after tokenisation. Returning empty results.")
            return []

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Sort descending by score
        n = len(self._doc_ids)
        ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)

        results = []
        rank = 1
        for idx in ranked_indices:
            if rank > top_k:
                break

            meta = self._doc_metas[idx]

            # Apply metadata filter (client-side, since BM25 has no built-in filter)
            if metadata_filter and not _matches_filter(meta, metadata_filter):
                continue

            results.append(
                {
                    "document_id":      self._doc_ids[idx],
                    "text":             self._doc_texts[idx],
                    "score":            float(scores[idx]),
                    "rank":             rank,
                    "retrieval_method": "bm25",
                    "metadata":         meta,
                }
            )
            rank += 1

        return results

    def get_ranked_ids(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> list[str]:
        """
        Return just the document IDs in BM25 rank order.
        Used by the RRF fusion step.
        """
        results = self.search(query, top_k=top_k)
        return [r["document_id"] for r in results]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches_filter(metadata: dict, filter_dict: dict) -> bool:
    """Return True if all key-value pairs in filter_dict match metadata."""
    for key, value in filter_dict.items():
        if metadata.get(key) != value:
            return False
    return True
