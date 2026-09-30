"""
embeddings.py - Sentence-Transformer embedding model wrapper.

We wrap the SentenceTransformer in a class so the model is loaded only
once (important for Streamlit caching) and shared across the application.

Preferred model: BAAI/bge-small-en-v1.5
  - 33 M parameters, ~130 MB
  - 384-dimensional embeddings
  - Excellent for English retrieval tasks

Fallback: sentence-transformers/all-MiniLM-L6-v2
  - 22 M parameters, ~90 MB
  - 384-dimensional embeddings
"""

import numpy as np
from typing import Optional

from src.config import BATCH_SIZE, EMBEDDING_MODEL
from src.utils import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wraps a SentenceTransformer to provide convenient encode methods.

    Usage
    -----
        model = EmbeddingModel()
        model.load()
        query_vec   = model.encode_query("What causes hallucinations?")
        doc_vecs    = model.encode_documents(["passage 1 …", "passage 2 …"])
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Download (first time) and load the embedding model into memory.
        Subsequent calls are no-ops.
        """
        if self._model is not None:
            return  # already loaded

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.  "
                "Install it with: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {self.model_name} …")
        try:
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded. Dimension = {self.embedding_dim}")
        except Exception as e:
            # Fallback to a well-known public model
            fallback = "sentence-transformers/all-MiniLM-L6-v2"
            logger.warning(
                f"Could not load '{self.model_name}': {e}. "
                f"Falling back to '{fallback}'."
            )
            self._model = SentenceTransformer(fallback)
            self.model_name = fallback
            logger.info(f"Fallback model loaded. Dimension = {self.embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of document texts into normalised embedding vectors.

        Parameters
        ----------
        texts       : list of str
        batch_size  : int – number of texts per GPU/CPU batch
        show_progress : bool – show tqdm progress bar

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        Vectors are L2-normalised so cosine similarity == dot product.
        """
        self.load()

        # BGE models need a prefix for document encoding
        prefix = ""
        if "bge" in self.model_name.lower():
            # BGE recommendation: add prefix for passages
            texts = [t for t in texts]  # keep originals; BGE handles via prompt

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalise → cosine = dot
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        BGE models recommend a special instruction prefix for queries.
        Returns a 1-D np.ndarray of shape (embedding_dim,).
        """
        self.load()

        # BGE instruction prefix for retrieval queries
        if "bge" in self.model_name.lower():
            query = "Represent this sentence for searching relevant passages: " + query

        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)


# ── Module-level singleton ────────────────────────────────────────────────────
# Other modules import this and call .load() once.
_default_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> EmbeddingModel:
    """
    Return a module-level singleton EmbeddingModel.
    The model is loaded on first call and reused thereafter.
    """
    global _default_model
    if _default_model is None or _default_model.model_name != model_name:
        _default_model = EmbeddingModel(model_name)
        _default_model.load()
    return _default_model
