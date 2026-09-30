"""
config.py - Central configuration for the Hybrid RAG Retrieval system.

All settings live here. You can override any value by setting an
environment variable with the same name (e.g. export DEMO_SIZE=5000).
"""

import os
from pathlib import Path

# ── Project root (the hybrid-rag-retrieval/ folder) ──────────────────────────
ROOT_DIR = Path(__file__).parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

# Processed passages file (written by preprocess_data.py)
PASSAGES_FILE   = PROCESSED_DIR / "passages.jsonl"

# ── Index paths ───────────────────────────────────────────────────────────────
INDEX_DIR       = ROOT_DIR / "indexes"
VECTOR_DB_PATH  = INDEX_DIR / "vector"          # ChromaDB persists here
BM25_INDEX_PATH = INDEX_DIR / "bm25" / "bm25_index.pkl"

# ── Embedding model ───────────────────────────────────────────────────────────
# BAAI/bge-small-en-v1.5 is fast and accurate. Falls back to all-MiniLM-L6-v2.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── Retrieval defaults ────────────────────────────────────────────────────────
DENSE_TOP_K  = int(os.getenv("DENSE_TOP_K",  "20"))   # candidates from vector DB
BM25_TOP_K   = int(os.getenv("BM25_TOP_K",   "20"))   # candidates from BM25
FINAL_TOP_K  = int(os.getenv("FINAL_TOP_K",  "5"))    # final results shown
RRF_K        = int(os.getenv("RRF_K",        "60"))   # RRF smoothing constant

# ── Dataset sizes ─────────────────────────────────────────────────────────────
DEMO_SIZE         = int(os.getenv("DEMO_SIZE",         "10000"))
FULL_DATASET_SIZE = int(os.getenv("FULL_DATASET_SIZE", "100000"))

# ── Embedding batch size ──────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# ── ChromaDB collection name ──────────────────────────────────────────────────
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ms_marco_passages")

# ── Benchmark settings ────────────────────────────────────────────────────────
BENCHMARK_QUERIES = int(os.getenv("BENCHMARK_QUERIES", "50"))

# ── Results output ────────────────────────────────────────────────────────────
RESULTS_DIR     = ROOT_DIR / "results"
EVAL_DIR        = RESULTS_DIR / "evaluation"
BENCHMARK_DIR   = RESULTS_DIR / "benchmarks"

# ── Reproducibility seed ──────────────────────────────────────────────────────
RANDOM_SEED = 42
