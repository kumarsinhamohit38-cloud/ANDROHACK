# Precision Retrieval for Reliable Enterprise RAG

A high-precision **hybrid information retrieval system** for enterprise RAG applications.
Combines **dense vector search + BM25 sparse retrieval + Reciprocal Rank Fusion (RRF)**
over the MS MARCO dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [Why Dense Retrieval Alone Is Insufficient](#4-why-dense-retrieval-alone-is-insufficient)
5. [Why BM25 Is Useful](#5-why-bm25-is-useful)
6. [What RRF Does](#6-what-rrf-does)
7. [Metadata Filtering](#7-metadata-filtering)
8. [Dataset Setup](#8-dataset-setup)
9. [Installation](#9-installation)
10. [Index Building](#10-index-building)
11. [Running the Streamlit App](#11-running-the-streamlit-app)
12. [Evaluation](#12-evaluation)
13. [Benchmarking](#13-benchmarking)
14. [Project Structure](#14-project-structure)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitations](#16-limitations)
17. [Future Improvements](#17-future-improvements)

---

## 1. Project Overview

This project builds a complete **hybrid retrieval pipeline** that:

- Indexes 10,000–100,000+ MS MARCO passages
- Provides three retrieval modes: Dense, BM25, and Hybrid RRF
- Returns Top-5 results with scores and explainability
- Measures retrieval quality (Recall@5, Precision@5, MRR, Context Precision, Context Recall)
- Measures actual query latency (targeting p95 < 300 ms)
- Provides a clean Streamlit web UI

---

## 2. Problem Statement

In enterprise RAG (Retrieval-Augmented Generation) systems, incorrect context retrieval
leads to **hallucinations** in the LLM's final answer.

The core challenges are:

| Problem | Impact |
|---------|--------|
| Dense search misses exact keywords | Relevant documents ranked too low |
| BM25 misses semantic meaning | Related documents missed |
| Either alone = incomplete retrieval | LLM receives wrong context |
| Wrong context → LLM hallucination | Inaccurate, unreliable answers |

**Solution:** Combine both approaches with RRF fusion to get the best of both worlds.

---

## 3. Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        DATA PIPELINE                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   MS MARCO Dataset                                               ║
║        │                                                         ║
║        ▼                                                         ║
║   Data Ingestion & Preprocessing                                 ║
║        │                                                         ║
║        ├──────────────────────────────────┐                      ║
║        │                                  │                      ║
║        ▼                                  ▼                      ║
║   HuggingFace Embedding Model        BM25 Tokenisation           ║
║   (BAAI/bge-small-en-v1.5)                │                      ║
║        │                                  │                      ║
║        ▼                                  ▼                      ║
║   ChromaDB Vector Index            BM25 Index (pickle)           ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                        QUERY LAYER                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   User Query                                                     ║
║        │                                                         ║
║        ├──────────────────────────────────┐                      ║
║        │                                  │                      ║
║        ▼                                  ▼                      ║
║   Dense Vector Search             BM25 Keyword Search            ║
║   (ChromaDB cosine sim)           (rank_bm25)                    ║
║        │                                  │                      ║
║        └──────────────┬───────────────────┘                      ║
║                       │                                          ║
║                       ▼                                          ║
║              Reciprocal Rank Fusion                              ║
║                       │                                          ║
║                       ▼                                          ║
║              Metadata Filtering                                  ║
║                       │                                          ║
║                       ▼                                          ║
║              Top-5 Retrieved Passages                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 4. Why Dense Retrieval Alone Is Insufficient

Dense retrieval (embedding-based cosine similarity) is excellent at:
- Semantic matching: "fever" ↔ "high temperature"
- Paraphrase recognition

But it can **fail** when:

| Scenario | Example |
|----------|---------|
| Exact ID/number needed | "RFC 2616" vs "HTTP specification" |
| Rare technical terms | "BM25Okapi" vs "ranking function" |
| Short exact queries | "p53 mutation" → dense may miss exact match |
| Vocabulary mismatch | Domain-specific jargon |

---

## 5. Why BM25 Is Useful

BM25 (Best Match 25) is a classical probabilistic ranking function that:

- Matches **exact keywords** from the query
- Accounts for **term frequency** (how often a word appears)
- Accounts for **document length** (longer docs penalised)
- Has **no semantic understanding** but strong lexical precision

BM25 excels where dense search fails (exact terms) and vice versa.

---

## 6. What RRF Does

**Reciprocal Rank Fusion** combines multiple ranked lists without needing to
normalise their scores (dense scores and BM25 scores are on completely different scales).

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
               over all ranked lists i
```

Where:
- `rank_i(doc)` = position of doc in list i (1-indexed)
- `k` = smoothing constant (default 60, from the original 2009 paper)

**Key insight:** A document ranked high in BOTH lists gets a much higher fused score.

Example:
```
Dense ranking: [A, B, C, D]
BM25 ranking:  [B, E, A, C]

RRF scores (k=60):
  A: 1/61 + 1/63 = 0.0321   (rank 1 dense + rank 3 BM25)
  B: 1/62 + 1/61 = 0.0325   (rank 2 dense + rank 1 BM25) ← WINS
  C: 1/63 + 1/64 = 0.0315
  D: 1/64         = 0.0156   (only in dense)
  E: 1/62         = 0.0161   (only in BM25)
```

B wins because it appeared high in both lists!

---

## 7. Metadata Filtering

The system supports filtering by:

| Field | Values | Effect |
|-------|--------|--------|
| `source` | `ms_marco` | Keep only MS MARCO passages |
| `dataset` | `ms_marco` | Same as above |
| `is_selected` | `True` | Keep only MS MARCO "relevant" passages |

**ChromaDB** applies pre-filtering server-side (before similarity search).
**BM25** applies post-filtering client-side (after scoring).

This reduces the search space and can improve precision in production systems
where passages from multiple domains are mixed.

---

## 8. Dataset Setup

We use the **MS MARCO Passage Ranking** dataset from HuggingFace.

```bash
# Demo mode: ~10,000 passages (fast, good for development)
python scripts/download_data.py --mode demo

# Full mode: ~100,000 passages
python scripts/download_data.py --mode full

# Custom size
python scripts/download_data.py --size 50000
```

The script:
1. Downloads MS MARCO via the HuggingFace `datasets` library
2. Extracts and deduplicates passages
3. Preprocesses (cleans text, tokenises for BM25)
4. Saves to `data/processed/passages.jsonl`

---

## 9. Installation

### Prerequisites
- Python 3.10+ (tested with 3.10, 3.11, 3.12, 3.14)
- 4+ GB disk space (for models + indexes)
- Internet connection (first run: downloads model + dataset)

### Steps

```bash
# 1. Clone or navigate to the project directory
cd hybrid-rag-retrieval

# 2. Create a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Using the existing venv (this repo)

```bash
# From the ANDOHACK directory:
.\venv\Scripts\activate

# Then cd into hybrid-rag-retrieval and run scripts
cd hybrid-rag-retrieval
```

---

## 10. Index Building

Run these commands in order:

```bash
# Step 1: Download and preprocess data (demo mode = 10k passages)
python scripts/download_data.py --mode demo

# Step 2: Build both vector (ChromaDB) and BM25 indexes
python scripts/build_indexes.py
```

This will:
1. Load the `BAAI/bge-small-en-v1.5` embedding model (~130 MB download on first run)
2. Encode all passages into 384-d vectors
3. Store vectors in ChromaDB (persistent, in `indexes/vector/`)
4. Build BM25 index (saved to `indexes/bm25/bm25_index.pkl`)

**Expected time:**
- 10k passages: ~5–15 minutes (depends on CPU/GPU)
- 100k passages: ~30–90 minutes

---

## 11. Running the Streamlit App

```bash
streamlit run app.py
```

Open in browser: http://localhost:8501

### Search your own files

In the app sidebar, add a PDF, Word document, CSV, or text/code file (including CSS).
The app extracts its readable text, splits it into searchable passages, and uses the
project's embedding model to find relevant passages. Select **Uploaded documents**
under **Search in**, then ask a question. The results are limited to the uploaded
files, so they do not mix with the main knowledge library.

Uploads are held in the current app session and are cleared when that session ends
or when **Clear uploaded files** is selected. PDFs must contain selectable text;
scanned-image PDFs are not processed because OCR is not enabled.

After installing the updated requirements with `pip install -r requirements.txt`,
restart the app. No dataset preprocessing or index rebuild is needed for uploads.

### Features

- **Single method search:** Dense, BM25, or Hybrid RRF
- **Comparison mode:** Compare all three side-by-side in tabs
- **Latency display:** Per-query timing including breakdown
- **Explainability:** For hybrid results, shows dense rank + BM25 rank + RRF score
- **Evaluation dashboard:** Shows metrics after running `evaluate.py`
- **Benchmark dashboard:** Shows p95 latency after running `benchmark.py`

---

## 12. Evaluation

```bash
python scripts/evaluate.py
# or
python scripts/evaluate.py --queries 200 --k 5
```

By default this evaluates queries whose labeled relevant passages are present in
the local MS MARCO index, runs all three retrievers, and computes:

| Metric | Description |
|--------|-------------|
| Recall@5 | Fraction of relevant passages found in top-5 |
| Precision@5 | Fraction of top-5 that are relevant |
| MRR | Mean Reciprocal Rank |
| Context Precision | RAGAS-style (= Precision@5 here) |
| Context Recall | RAGAS-style (= Recall@5 here) |

**Note on RAGAS:** Full RAGAS metrics require an LLM API for context faithfulness
scoring. We implement equivalent deterministic retrieval metrics instead and
document this clearly. The metrics measure the same underlying concepts.

Results are saved to `results/evaluation/evaluation_results.json`.

To measure the deliberately harder cross-split corpus-mismatch case, use:

```bash
python scripts/evaluate.py --query-source validation
```

---

## 13. Benchmarking

```bash
python scripts/benchmark.py
# or
python scripts/benchmark.py --queries 100
```

Measures real latency (not simulated) across 50+ queries.

Reports:
- Mean, median, p50, p95, p99 latency
- Comparison table
- Target: p95 < 300 ms (realistic on most modern machines with ChromaDB)

Results are saved to `results/benchmarks/benchmark_results.json`.

---

## 14. Project Structure

```
hybrid-rag-retrieval/
│
├── app.py                    ← Streamlit UI
├── README.md                 ← This file
├── requirements.txt          ← Python dependencies
├── .gitignore
├── .env.example              ← Configuration reference
│
├── data/
│   ├── raw/                  ← Downloaded dataset files (auto-created)
│   ├── processed/
│   │   └── passages.jsonl    ← Preprocessed passages (created by download_data.py)
│   └── README.md
│
├── indexes/
│   ├── vector/               ← ChromaDB persistent storage
│   └── bm25/
│       └── bm25_index.pkl    ← BM25 index
│
├── src/
│   ├── __init__.py
│   ├── config.py             ← Central configuration
│   ├── data_loader.py        ← MS MARCO dataset loading
│   ├── preprocessing.py      ← Text cleaning & tokenisation
│   ├── embeddings.py         ← Sentence-transformer wrapper
│   ├── vector_store.py       ← ChromaDB vector store
│   ├── bm25_search.py        ← BM25 retriever
│   ├── dense_search.py       ← Dense retriever
│   ├── rrf.py                ← Reciprocal Rank Fusion
│   ├── hybrid_search.py      ← Hybrid retriever (Dense + BM25 + RRF)
│   ├── metadata_filter.py    ← Metadata filtering utilities
│   ├── evaluation.py         ← Evaluation pipeline
│   ├── metrics.py            ← IR metrics (Recall, Precision, MRR)
│   └── utils.py              ← Shared utilities
│
├── scripts/
│   ├── download_data.py      ← Download + preprocess MS MARCO
│   ├── preprocess_data.py    ← Re-preprocess without re-downloading
│   ├── build_indexes.py      ← Build vector + BM25 indexes
│   ├── evaluate.py           ← Run evaluation
│   └── benchmark.py          ← Run latency benchmarks
│
├── tests/
│   ├── test_rrf.py           ← RRF unit tests
│   ├── test_bm25.py          ← BM25 unit tests
│   ├── test_hybrid.py        ← Hybrid + metadata filter tests
│   └── test_pipeline.py      ← Preprocessing + metrics tests
│
└── results/
    ├── evaluation/           ← evaluation_results.json
    └── benchmarks/           ← benchmark_results.json
```

---

## 15. Troubleshooting

### "Passages file not found"
Run: `python scripts/download_data.py --mode demo`

### "BM25 index not found"
Run: `python scripts/build_indexes.py`

### "Vector store is empty"
Run: `python scripts/build_indexes.py`

### Embedding model download fails
Check your internet connection. The model downloads from HuggingFace (~130 MB).
It's cached after first download.

### ChromaDB import error
Run: `pip install chromadb`

### datasets import error
Run: `pip install datasets`

### Streamlit shows blank screen
Check that you're running from inside `hybrid-rag-retrieval/` directory.

### Out of memory during indexing
Reduce `BATCH_SIZE` in `.env`: `BATCH_SIZE=16`

### Slow indexing
Normal on CPU. 10k passages takes ~5–15 min. Use `--mode demo` for testing.

---

## 16. Limitations

1. **Evaluation is deterministic, not LLM-based:** Full RAGAS requires an LLM API.
   We implement equivalent IR metrics (Recall@K, Precision@K, MRR) which are
   industry-standard and do not require an LLM.

2. **Demo mode:** With only 10k passages, many evaluation queries may not find
   their specific relevant passage. Larger datasets improve recall significantly.

3. **No GPU acceleration by default:** Embedding on CPU is slower. With a GPU,
   expect 10–100x speedup.

4. **ChromaDB HNSW limitations:** Very large datasets (1M+) may require Qdrant
   or a dedicated vector database.

5. **BM25 is exact-match only:** Typos or morphological variants may reduce BM25
   recall.

6. **No re-ranking:** A cross-encoder re-ranker would further improve precision.

---

## 17. Future Improvements

- [ ] Cross-encoder re-ranking (e.g., `ms-marco-MiniLM-L-6-v2`)
- [ ] GPU-accelerated embedding
- [ ] Qdrant backend for larger scale
- [ ] Query expansion (HyDE, query rewriting)
- [ ] Learning-to-rank with MS MARCO relevance labels
- [ ] Full RAGAS integration with optional LLM (OpenAI / local Ollama)
- [ ] Multi-modal retrieval
- [ ] Streaming results in the UI
- [ ] Docker containerisation

---

## Quick Reference

```bash
# Full workflow:
python scripts/download_data.py --mode demo
python scripts/build_indexes.py
streamlit run app.py

# Tests:
pytest tests/ -v

# Evaluation:
python scripts/evaluate.py --queries 100

# Benchmarks:
python scripts/benchmark.py --queries 50
```
