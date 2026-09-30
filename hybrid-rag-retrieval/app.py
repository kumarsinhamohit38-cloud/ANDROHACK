"""
app.py - Streamlit UI for Precision Retrieval for Reliable Enterprise RAG.

Run with:  streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ── Make 'src' importable ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DENSE_TOP_K, BM25_TOP_K, FINAL_TOP_K, RRF_K,
    PASSAGES_FILE, BM25_INDEX_PATH, VECTOR_DB_PATH,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precision Retrieval for Enterprise RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(99,102,241,0.15);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a5b4fc;
}
.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.result-card {
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #0f172a;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: #6366f1;
}
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
}
.method-badge-dense {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
    color: white;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 500;
}
.method-badge-bm25 {
    background: linear-gradient(135deg, #f59e0b, #f97316);
    color: white;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 500;
}
.method-badge-hybrid {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 500;
}
.setup-warning {
    background: linear-gradient(135deg, #1c1917 0%, #292524 100%);
    border: 1px solid #78350f;
    border-radius: 10px;
    padding: 1.5rem;
    color: #fcd34d;
}
.stExpander > div { border: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ───────────────────────────────────────────────────
# @st.cache_resource ensures these are loaded ONCE and reused across queries.

@st.cache_resource(show_spinner="Loading embedding model …")
def load_embedding_model():
    from src.embeddings import get_embedding_model
    return get_embedding_model()


@st.cache_resource(show_spinner="Loading vector store …")
def load_vector_store():
    from src.vector_store import VectorStore
    store = VectorStore()
    store.create_collection()
    return store


@st.cache_resource(show_spinner="Loading BM25 index …")
def load_bm25_retriever():
    from src.bm25_search import BM25Retriever
    retriever = BM25Retriever()
    retriever.load()
    return retriever


# ── Helper: check if indexes exist ────────────────────────────────────────────

def check_setup() -> tuple[bool, list[str]]:
    """Return (all_ready, list_of_missing_items)."""
    missing = []
    if not PASSAGES_FILE.exists():
        missing.append("Passages file not found. Run: `python scripts/download_data.py`")
    if not BM25_INDEX_PATH.exists():
        missing.append("BM25 index not found. Run: `python scripts/build_indexes.py`")
    # ChromaDB: check if the directory has any content
    vector_dir = Path(VECTOR_DB_PATH)
    if not vector_dir.exists() or not any(vector_dir.iterdir()):
        missing.append("Vector index not found. Run: `python scripts/build_indexes.py`")
    return len(missing) == 0, missing


# ── Helper: render a single result card ──────────────────────────────────────

def render_result_card(result: dict, idx: int):
    """Render one retrieved passage as an expander card."""
    method     = result.get("retrieval_method", "unknown")
    score      = result.get("score", 0.0)
    rank       = result.get("rank", idx)
    doc_id     = result.get("document_id", "unknown")
    text       = result.get("text", "")
    meta       = result.get("metadata", {})

    method_class = f"method-badge-{method}"

    label = (
        f"🔹 Rank {rank} &nbsp;|&nbsp; "
        f"Score: {score:.4f} &nbsp;|&nbsp; "
        f"<span class='{method_class}'>{method.upper()}</span> &nbsp;|&nbsp; "
        f"ID: {doc_id}"
    )

    with st.expander(f"Rank {rank} — {doc_id}  (score: {score:.4f})", expanded=(idx == 0)):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**📄 Passage Text**")
            st.markdown(f"> {text[:800]}{'…' if len(text) > 800 else ''}")
        with col2:
            st.markdown("**📊 Retrieval Info**")
            st.markdown(f"- **Method:** `{method}`")
            st.markdown(f"- **Score:** `{score:.5f}`")
            st.markdown(f"- **Rank:** `{rank}`")
            st.markdown(f"- **Doc ID:** `{doc_id}`")

            # Hybrid-specific explainability
            if method == "hybrid":
                st.markdown("---")
                st.markdown("**🔀 Fusion Details**")
                dense_rank = result.get("dense_rank")
                bm25_rank  = result.get("bm25_rank")
                rrf_score  = result.get("rrf_score", score)
                st.markdown(f"- **RRF Score:** `{rrf_score:.6f}`")
                st.markdown(
                    f"- **Dense Rank:** `{dense_rank if dense_rank else 'not retrieved'}`"
                )
                st.markdown(
                    f"- **BM25 Rank:** `{bm25_rank if bm25_rank else 'not retrieved'}`"
                )

            # Metadata
            if meta:
                st.markdown("---")
                st.markdown("**🏷️ Metadata**")
                for k, v in list(meta.items())[:6]:
                    st.markdown(f"- **{k}:** `{v}`")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-title">🔍 Precision Retrieval for Reliable Enterprise RAG</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Hybrid Dense + BM25 Retrieval with Reciprocal Rank Fusion</div>',
        unsafe_allow_html=True,
    )

    # ── Setup check ───────────────────────────────────────────────────────────
    all_ready, missing = check_setup()

    if not all_ready:
        st.markdown('<div class="setup-warning">', unsafe_allow_html=True)
        st.warning("⚠️  **Setup required** — Some components are not ready yet.")
        for m in missing:
            st.code(m)
        st.markdown("**Quick setup:**")
        st.code(
            "python scripts/download_data.py --mode demo\n"
            "python scripts/build_indexes.py",
            language="bash",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.divider()

        retrieval_method = st.selectbox(
            "Retrieval Method",
            options=["Hybrid RRF", "Dense Baseline", "BM25"],
            index=0,
            help=(
                "Hybrid RRF = dense + BM25 fused with RRF (best quality)\n"
                "Dense = vector similarity search\n"
                "BM25 = keyword frequency search"
            ),
        )

        st.divider()
        st.markdown("### 🎚️ Parameters")

        final_top_k = st.slider("Top-K Results", min_value=1, max_value=10, value=5)
        dense_top_k = st.slider("Dense Candidates", min_value=5, max_value=50, value=DENSE_TOP_K)
        bm25_top_k  = st.slider("BM25 Candidates",  min_value=5, max_value=50, value=BM25_TOP_K)
        rrf_k       = st.slider("RRF K (smoothing)", min_value=1, max_value=120, value=RRF_K)

        st.divider()
        st.markdown("### 🔽 Metadata Filters")
        filter_source = st.selectbox(
            "Source Filter",
            options=["All", "ms_marco"],
            index=0,
        )
        filter_selected = st.selectbox(
            "Relevance Filter",
            options=["All", "Relevant Only (is_selected=True)"],
            index=0,
        )

        st.divider()
        st.markdown("### 📊 Comparison Mode")
        compare_mode = st.toggle("Compare All Methods", value=False)

        st.divider()
        st.caption("💡 Data: MS MARCO passages")
        st.caption("💡 Embeddings: BAAI/bge-small-en-v1.5")
        st.caption("💡 Vector DB: ChromaDB")

    # ── Load resources (cached) ───────────────────────────────────────────────
    embedding_model = load_embedding_model()
    vector_store    = load_vector_store()
    bm25_retriever  = load_bm25_retriever()

    from src.dense_search  import DenseRetriever
    from src.hybrid_search import HybridRetriever

    dense_retriever  = DenseRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        rrf_k=rrf_k,
    )

    # Build metadata filter
    metadata_filter = {}
    if filter_source == "ms_marco":
        metadata_filter["source"] = "ms_marco"
    if filter_selected == "Relevant Only (is_selected=True)":
        metadata_filter["is_selected"] = True
    if not metadata_filter:
        metadata_filter = None

    # ── Main search interface ─────────────────────────────────────────────────
    st.markdown("## 🔎 Search")

    query = st.text_input(
        "Enter your query",
        placeholder="What causes hallucinations in enterprise RAG systems?",
        label_visibility="collapsed",
    )

    search_col, clear_col = st.columns([1, 4])
    with search_col:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    # ── Comparison mode ───────────────────────────────────────────────────────
    if compare_mode and query and search_clicked:
        st.markdown("---")
        st.markdown("## 📊 Side-by-Side Comparison")

        tabs = st.tabs(["🔵 Dense", "🟡 BM25", "🟢 Hybrid RRF"])

        def run_dense():
            t0 = time.perf_counter()
            results = dense_retriever.search(query, top_k=final_top_k, metadata_filter=metadata_filter)
            return results, (time.perf_counter() - t0) * 1000

        def run_bm25():
            t0 = time.perf_counter()
            results = bm25_retriever.search(query, top_k=final_top_k, metadata_filter=metadata_filter)
            return results, (time.perf_counter() - t0) * 1000

        def run_hybrid():
            t0 = time.perf_counter()
            response = hybrid_retriever.search(query, metadata_filter=metadata_filter)
            return response["results"], (time.perf_counter() - t0) * 1000, response["latency"]

        with st.spinner("Running all three retrievers …"):
            dense_results, dense_ms  = run_dense()
            bm25_results,  bm25_ms   = run_bm25()
            hybrid_results, hybrid_ms, hybrid_latency = run_hybrid()

        with tabs[0]:
            st.metric("Latency", f"{dense_ms:.1f} ms")
            st.metric("Results", len(dense_results))
            for i, r in enumerate(dense_results):
                render_result_card(r, i + 1)

        with tabs[1]:
            st.metric("Latency", f"{bm25_ms:.1f} ms")
            st.metric("Results", len(bm25_results))
            for i, r in enumerate(bm25_results):
                render_result_card(r, i + 1)

        with tabs[2]:
            st.metric("Latency", f"{hybrid_ms:.1f} ms")
            st.metric("Results", len(hybrid_results))
            with st.expander("⏱️ Latency Breakdown"):
                for k_name, v in hybrid_latency.items():
                    st.text(f"  {k_name}: {v:.2f} ms")
            for i, r in enumerate(hybrid_results):
                render_result_card(r, i + 1)

        return  # Don't show the single-method results below

    # ── Single method search ──────────────────────────────────────────────────
    if not (query and search_clicked):
        if not query:
            st.info(
                "👆 Enter a search query above and click **Search**.\n\n"
                "**Example queries:**\n"
                "- What causes hallucinations in language models?\n"
                "- How does BM25 keyword retrieval work?\n"
                "- What is reciprocal rank fusion?\n"
                "- How do vaccines prevent disease?"
            )
        return

    st.markdown("---")

    # Run retrieval
    with st.spinner(f"Searching with {retrieval_method} …"):
        try:
            t0 = time.perf_counter()

            if retrieval_method == "Dense Baseline":
                results = dense_retriever.search(
                    query, top_k=final_top_k, metadata_filter=metadata_filter
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                latency_breakdown = {"total_ms": latency_ms}

            elif retrieval_method == "BM25":
                results = bm25_retriever.search(
                    query, top_k=final_top_k, metadata_filter=metadata_filter
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                latency_breakdown = {"total_ms": latency_ms}

            else:  # Hybrid RRF
                response = hybrid_retriever.search(query, metadata_filter=metadata_filter)
                results           = response["results"]
                latency_breakdown = response["latency"]
                latency_ms        = latency_breakdown.get("total_ms", 0.0)

        except Exception as e:
            st.error(f"❌ Retrieval failed: {e}")
            st.exception(e)
            return

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown("## 📈 Query Metrics")

    top_score = results[0]["score"] if results else 0.0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🕐 Latency", f"{latency_ms:.1f} ms",
                  delta="✅ <300ms" if latency_ms < 300 else "⚠️ >300ms")
    with m2:
        st.metric("📋 Method", retrieval_method)
    with m3:
        st.metric("🏆 Results", len(results))
    with m4:
        st.metric("🥇 Top Score", f"{top_score:.4f}")

    # Latency breakdown for hybrid
    if retrieval_method == "Hybrid RRF" and latency_breakdown:
        with st.expander("⏱️ Detailed Latency Breakdown"):
            cols = st.columns(len(latency_breakdown))
            for col, (key, val) in zip(cols, latency_breakdown.items()):
                col.metric(key, f"{val:.2f} ms")

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"## 📄 Top-{len(results)} Retrieved Passages")

    if not results:
        st.warning("No results found. Try a different query or check the metadata filters.")
        return

    for i, result in enumerate(results):
        render_result_card(result, i + 1)

    # ── Evaluation dashboard ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🧪 Evaluation Dashboard")

    eval_file = Path("results/evaluation/evaluation_results.json")
    if eval_file.exists():
        import json
        with open(eval_file) as f:
            eval_data = json.load(f)

        col_d, col_b, col_h = st.columns(3)
        methods = [("Dense", "dense", col_d), ("BM25", "bm25", col_b), ("Hybrid", "hybrid", col_h)]

        for label, key, col in methods:
            m = eval_data.get(key, {})
            with col:
                st.markdown(f"**{label}**")
                st.metric("Recall@K",    f"{m.get('recall_at_k', 0):.4f}")
                st.metric("Precision@K", f"{m.get('precision_at_k', 0):.4f}")
                st.metric("MRR",         f"{m.get('mrr', 0):.4f}")
                st.metric("Ctx Precision", f"{m.get('context_precision', 0):.4f}")
                st.metric("Ctx Recall",    f"{m.get('context_recall', 0):.4f}")

        if "table" in eval_data:
            with st.expander("📊 Full Comparison Table"):
                st.code(eval_data["table"])
    else:
        st.info(
            "No evaluation results yet.\n\n"
            "Run:  `python scripts/evaluate.py`"
        )

    # Benchmark results
    benchmark_file = Path("results/benchmarks/benchmark_results.json")
    if benchmark_file.exists():
        import json
        with open(benchmark_file) as f:
            bench_data = json.load(f)

        st.markdown("### ⚡ Benchmark Latency")
        cols = st.columns(3)
        for (label, key), col in zip(
            [("Dense", "dense"), ("BM25", "bm25"), ("Hybrid", "hybrid")], cols
        ):
            b = bench_data.get(key, {})
            with col:
                st.markdown(f"**{label}**")
                st.metric("p95 Latency", f"{b.get('p95_ms', 0):.1f} ms",
                          delta="✅" if b.get("p95_ms", 999) < 300 else "⚠️")
                st.metric("Mean", f"{b.get('mean_ms', 0):.1f} ms")
                st.metric("p99",  f"{b.get('p99_ms', 0):.1f} ms")
    else:
        st.info(
            "No benchmark results yet.\n\n"
            "Run:  `python scripts/benchmark.py`"
        )


if __name__ == "__main__":
    main()
