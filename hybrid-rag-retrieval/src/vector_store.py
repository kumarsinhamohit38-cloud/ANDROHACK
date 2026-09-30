"""
vector_store.py - ChromaDB-based persistent vector store.

ChromaDB runs entirely in-process with no server required.
Passage vectors and metadata are stored on disk at VECTOR_DB_PATH.

Why ChromaDB?
- Zero server setup (unlike Qdrant server mode)
- Persistent storage across restarts
- Supports metadata filtering
- Simple Python API
"""

from pathlib import Path
from typing import Optional

import numpy as np

from src.config import CHROMA_COLLECTION, FINAL_TOP_K, VECTOR_DB_PATH
from src.utils import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Wraps ChromaDB for passage storage and similarity search.

    Usage
    -----
        store = VectorStore()
        store.create_collection()
        store.insert(passages, embeddings)   # one-time indexing step
        results = store.search(query_vec, top_k=20)
    """

    def __init__(
        self,
        persist_path: str | Path = VECTOR_DB_PATH,
        collection_name: str = CHROMA_COLLECTION,
    ):
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Connection & collection
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily initialise the ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError(
                    "chromadb is required.  Install: pip install chromadb"
                )
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_path))
        return self._client

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create (or open) the ChromaDB collection.

        Parameters
        ----------
        recreate : bool
            If True, delete the existing collection first (reindex from scratch).
        """
        client = self._get_client()

        if recreate:
            try:
                client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection '{self.collection_name}'.")
            except Exception:
                pass  # Collection may not exist yet

        # get_or_create is idempotent
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            # ChromaDB's built-in distance is 'l2'; we'll use 'cosine'
            # so that higher score = more similar.
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Collection '{self.collection_name}' ready. "
            f"Count = {self._collection.count()}"
        )

    def _ensure_collection(self):
        if self._collection is None:
            self.create_collection()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(
        self,
        passages: list[dict],
        embeddings: np.ndarray,
        batch_size: int = 1000,
    ) -> None:
        """
        Insert passages and their embeddings into the vector store.

        Parameters
        ----------
        passages   : list of preprocessed passage dicts
                     (must have 'id', 'text', 'metadata')
        embeddings : np.ndarray of shape (N, dim)
        batch_size : int – ChromaDB batch size for upserts
        """
        self._ensure_collection()

        if len(passages) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(passages)} passages vs {len(embeddings)} embeddings"
            )

        logger.info(f"Inserting {len(passages)} passages into ChromaDB …")

        total = len(passages)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_passages = passages[start:end]
            batch_embeds   = embeddings[start:end]

            ids        = [p["id"] for p in batch_passages]
            documents  = [p["text"] for p in batch_passages]
            metadatas  = []
            for p in batch_passages:
                # ChromaDB metadata values must be str/int/float/bool
                m = {}
                for k, v in p.get("metadata", {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        m[k] = v
                    else:
                        m[k] = str(v)
                metadatas.append(m)

            self._collection.upsert(
                ids=ids,
                embeddings=batch_embeds.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"  Inserted batch {start}–{end - 1}")

        logger.info("Insertion complete.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = FINAL_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Find the top_k most similar passages for a query vector.

        Parameters
        ----------
        query_vector    : 1-D np.ndarray (embedding_dim,)
        top_k           : number of results to return
        metadata_filter : optional ChromaDB 'where' clause dict, e.g.
                          {"source": "ms_marco"}

        Returns
        -------
        List of result dicts:
            document_id, text, score, rank, retrieval_method, metadata
        """
        self._ensure_collection()

        if self._collection.count() == 0:
            raise RuntimeError(
                "The vector store is empty.  "
                "Please run: python scripts/build_indexes.py"
            )

        query_list = query_vector.tolist()

        kwargs = {
            "query_embeddings": [query_list],
            "n_results":        min(top_k, self._collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter

        results = self._collection.query(**kwargs)

        # ChromaDB returns distance (0 = identical). Convert to similarity.
        # For cosine space: similarity = 1 - distance
        ids        = results["ids"][0]
        distances  = results["distances"][0]
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]

        output = []
        for rank, (doc_id, dist, text, meta) in enumerate(
            zip(ids, distances, documents, metadatas), start=1
        ):
            output.append(
                {
                    "document_id":       doc_id,
                    "text":              text,
                    "score":             float(1.0 - dist),   # cosine similarity
                    "rank":              rank,
                    "retrieval_method":  "dense",
                    "metadata":          meta,
                }
            )
        return output

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of passages indexed."""
        self._ensure_collection()
        return self._collection.count()

    def is_ready(self) -> bool:
        """Return True if the collection exists and has documents."""
        try:
            return self.count() > 0
        except Exception:
            return False

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """
        Fetch full passage information for a list of document IDs.
        Used by HybridRetriever to hydrate RRF results.
        """
        self._ensure_collection()
        if not ids:
            return []

        results = self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        output = []
        for doc_id, text, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            output.append(
                {
                    "document_id": doc_id,
                    "text":        text,
                    "metadata":    meta,
                }
            )
        return output
