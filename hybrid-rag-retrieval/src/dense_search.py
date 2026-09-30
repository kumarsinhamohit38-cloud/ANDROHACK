"""
dense_search.py - Dense vector retrieval using ChromaDB + sentence-transformers.

DenseRetriever ties together the EmbeddingModel and VectorStore into a
single object that encodes a query and searches the vector database.
"""

from typing import Optional

import numpy as np

from src.config import DENSE_TOP_K, EMBEDDING_MODEL
from src.embeddings import EmbeddingModel, get_embedding_model
from src.vector_store import VectorStore
from src.utils import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Dense passage retrieval.

    Query → embedding → cosine similarity search in ChromaDB → top-k results.

    Usage
    -----
        retriever = DenseRetriever()
        results = retriever.search("What causes fever?", top_k=20)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        # Use provided objects or create defaults
        self._store  = vector_store  or VectorStore()
        self._model  = embedding_model or get_embedding_model()

    # ------------------------------------------------------------------
    # Lazy init helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self):
        """Make sure the model is loaded and vector store collection exists."""
        self._model.load()
        if self._store._collection is None:
            self._store.create_collection()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve the top-k most semantically similar passages.

        Parameters
        ----------
        query           : str – the user's natural-language query
        top_k           : int – number of candidate passages to retrieve
        metadata_filter : optional dict for ChromaDB 'where' filtering,
                          e.g. {"source": "ms_marco"}

        Returns
        -------
        List of result dicts:
            document_id, text, score, rank, retrieval_method, metadata
        """
        if not query.strip():
            logger.warning("Empty query. Returning empty results.")
            return []

        self._ensure_ready()

        if not self._store.is_ready():
            raise RuntimeError(
                "Vector store is empty.  "
                "Run: python scripts/build_indexes.py"
            )

        # Encode the query to a vector
        query_vec = self._model.encode_query(query)

        # Search ChromaDB
        results = self._store.search(
            query_vector=query_vec,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return results

    def get_ranked_ids(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
    ) -> list[str]:
        """
        Return just the document IDs in dense-similarity rank order.
        Used by the RRF fusion step.
        """
        results = self.search(query, top_k=top_k)
        return [r["document_id"] for r in results]

    def encode_query(self, query: str) -> np.ndarray:
        """Expose query encoding (useful for latency benchmarking)."""
        self._model.load()
        return self._model.encode_query(query)
