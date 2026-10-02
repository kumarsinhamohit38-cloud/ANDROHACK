"""
app.py - Streamlit UI for Precision Retrieval for Reliable Enterprise RAG.

Run with:  streamlit run app.py
"""

import sys
import time
import json
import html
import re
import sqlite3
from pathlib import Path

import streamlit as st

# ── Make 'src' importable ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DENSE_TOP_K, BM25_TOP_K, FINAL_TOP_K, RRF_K,
    PASSAGES_FILE, BM25_INDEX_PATH, VECTOR_DB_PATH,
)
from src.answering import compose_grounded_answer

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Findwise | Knowledge Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clear Blue theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { --ink:#182a40; --muted:#718096; --blue:#3168d5; --line:#e1e9f3; --soft:#f3f7fd; }
.stApp { background:#f6f9fe; color:var(--ink); }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--ink); }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
.main-title { font-size:2.2rem; font-weight:750; color:var(--ink); margin:0 0 .2rem; }
.sub-title { color:var(--muted); font-size:1rem; margin:0 0 1.4rem; }
.brand-mark { display:flex; align-items:center; gap:.65rem; font-weight:750; font-size:1.15rem; color:var(--ink); padding:.25rem 0 .9rem; }
.brand-icon { width:34px; height:34px; display:inline-flex; align-items:center; justify-content:center; border-radius:10px; background:#3168d5; color:#fff; font-size:1.2rem; }
.eyebrow { color:#70819a; font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; margin-bottom:.5rem; }
.hero-title { color:var(--ink); font-size:2rem; line-height:1.18; font-weight:750; letter-spacing:-.035em; margin:.2rem 0; }
.hero-copy { color:var(--muted); margin:0 0 1.1rem; }
.answer-card { background:linear-gradient(135deg,#f0f6ff,#f8fbff); border:1px solid #d9e6fb; border-radius:16px; padding:1.35rem 1.5rem; margin:.75rem 0 1rem; box-shadow:0 8px 28px rgba(49,104,213,.06); }
.answer-label { display:flex; align-items:center; justify-content:space-between; gap:1rem; color:#244d96; font-size:.83rem; font-weight:700; margin-bottom:.7rem; }
.answer-text { color:#26384e; font-size:1.02rem; line-height:1.75; }
.source-card { background:#fff; border:1px solid var(--line); border-radius:13px; padding:1rem 1.15rem; margin:.65rem 0; transition:border-color .15s,box-shadow .15s; }
.source-card:hover { border-color:#b5c9ee; box-shadow:0 5px 18px rgba(30,60,100,.06); }
.source-head { display:flex; align-items:center; justify-content:space-between; gap:.8rem; color:var(--ink); font-weight:700; margin-bottom:.4rem; }
.source-meta { display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; color:#66768c; font-size:.78rem; margin-top:.7rem; }
.pill { display:inline-block; border-radius:99px; padding:.22rem .58rem; color:#2859ad; background:#eaf1ff; font-size:.72rem; font-weight:700; }
div.stButton > button[kind="primary"] { background:#3168d5; border:1px solid #3168d5; border-radius:10px; font-weight:650; }
div.stButton > button { border-radius:9px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:12px; }
div[data-baseweb="input"] input { background:#fff; }
hr { border-color:var(--line); }
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
    """Return readiness and explain missing or stale search indexes."""
    missing = []
    if not PASSAGES_FILE.exists():
        missing.append("Processed passages are missing. Add data/raw/passages_raw.jsonl and run scripts/preprocess_data.py.")
        return False, missing

    passages_mtime = PASSAGES_FILE.stat().st_mtime
    if not BM25_INDEX_PATH.exists():
        missing.append("Keyword index is missing. Run: `python scripts/build_indexes.py --recreate`.")
    elif BM25_INDEX_PATH.stat().st_mtime < passages_mtime:
        missing.append(
            "Keyword index is older than the combined dataset. Stop the app and run "
            "`python scripts/build_indexes.py --only bm25`, then restart the app."
        )
    # Check the actual vector record count. A recent SQLite timestamp alone
    # cannot prove that indexing finished; Streamlit may touch the database.
    vector_dir = Path(VECTOR_DB_PATH)
    sqlite_path = vector_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        missing.append("Semantic index is missing. Run: `python scripts/build_indexes.py --recreate`.")
    else:
        try:
            expected_count = sum(1 for line in PASSAGES_FILE.open(encoding="utf-8") if line.strip())
            with sqlite3.connect(str(sqlite_path)) as connection:
                indexed_count = connection.execute(
                    """SELECT COUNT(*) FROM embeddings e
                       JOIN segments s ON s.id = e.segment_id
                       JOIN collections c ON c.id = s.collection
                       WHERE c.name = ? AND s.scope = 'METADATA'""",
                    ("ms_marco_passages",),
                ).fetchone()[0]
            if indexed_count != expected_count:
                missing.append(
                    f"Semantic index contains {indexed_count} of {expected_count} passages. "
                    "Stop the app and run `python scripts/build_indexes.py --recreate`; "
                    "wait for **Indexes ready!**, then restart the app."
                )
        except (sqlite3.Error, OSError) as exc:
            missing.append(
                f"Could not verify the semantic index ({exc}). Stop the app and run "
                "`python scripts/build_indexes.py --recreate`, then restart the app."
            )
    return len(missing) == 0, missing


# ── Helper: render a single result card ──────────────────────────────────────

def render_source_card(result: dict, idx: int, show_scores: bool = True):
    """Render one safely escaped, source-backed passage card."""
    meta = result.get("metadata", {}) or {}
    text = html.escape(str(result.get("text", "")))
    title = html.escape(str(meta.get("topic") or meta.get("title") or result.get("document_id", "Source")))
    source = html.escape(str(meta.get("library") or meta.get("dataset") or meta.get("source") or "Library"))
    rank = result.get("rank", idx)
    score = result.get("score", 0.0)
    score_text = f" · Score {score:.4f}" if show_scores else ""
    st.markdown(
        f"""<div class="source-card">
        <div class="source-head"><span>{idx:02d} &nbsp; {title}</span>
        <span class="pill">{source} · Rank {rank}</span></div>
        <div>{text}</div>
        <div class="source-meta">{html.escape(str(result.get('retrieval_method', 'search')).upper())}{score_text}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def load_saved_searches() -> list[str]:
    """Read saved queries from the project's data folder."""
    saved_file = Path(__file__).parent / "data" / "saved_searches.json"
    try:
        values = json.loads(saved_file.read_text(encoding="utf-8"))
        return [str(value) for value in values if str(value).strip()]
    except (OSError, json.JSONDecodeError, TypeError):
        return []


def save_search(query: str) -> None:
    """Persist a query so it remains available after restarting the app."""
    query = query.strip()
    saved = load_saved_searches()
    if query and query.casefold() not in {item.casefold() for item in saved}:
        saved.insert(0, query)
        saved_file = Path(__file__).parent / "data" / "saved_searches.json"
        saved_file.parent.mkdir(parents=True, exist_ok=True)
        saved_file.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state.saved_searches = load_saved_searches()


def remove_saved_search(query: str) -> None:
    saved = [item for item in load_saved_searches() if item != query]
    saved_file = Path(__file__).parent / "data" / "saved_searches.json"
    saved_file.parent.mkdir(parents=True, exist_ok=True)
    saved_file.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state.saved_searches = saved


def open_search(query: str) -> None:
    st.session_state.query_input = query
    st.session_state.active_page = "Search"
    st.session_state.run_loaded_query = True


def get_metadata_values(field: str) -> list[str]:
    """Find filter values present in the processed corpus."""
    if not PASSAGES_FILE.exists():
        return []
    try:
        values = set()
        with PASSAGES_FILE.open(encoding="utf-8") as source_file:
            for line in source_file:
                if line.strip():
                    value = json.loads(line).get("metadata", {}).get(field)
                    if value is not None:
                        values.add(str(value))
        return sorted(values, key=str.casefold)
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.session_state.setdefault("active_page", "Search")
    st.session_state.setdefault("search_history", [])
    st.session_state.setdefault("saved_searches", load_saved_searches())
    st.session_state.setdefault("query_input", st.session_state.get("last_query", ""))

    all_ready, missing = check_setup()
    source_values = get_metadata_values("source")
    dataset_values = get_metadata_values("dataset")

    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><span class="brand-icon">⌕</span>Findwise</div>',
            unsafe_allow_html=True,
        )
        st.caption("Search your knowledge library")
        st.markdown("##### WORKSPACE")
        active_page = st.radio(
            "Navigation",
            ["Search", "History", "Saved"],
            key="active_page",
            label_visibility="collapsed",
        )
        retrieval_method = st.selectbox(
            "Search method",
            ["Hybrid RRF", "Dense Search", "BM25 Keyword"],
            help="Hybrid combines semantic and keyword matching. Dense finds related meaning; BM25 favors exact terms.",
        )
        with st.expander("Advanced search options", expanded=False):
            final_top_k = st.slider("Results to show", 1, 10, FINAL_TOP_K)
            dense_top_k = st.slider("Semantic candidates", 5, 50, DENSE_TOP_K)
            bm25_top_k = st.slider("Keyword candidates", 5, 50, BM25_TOP_K)
            rrf_k = st.slider("Hybrid balance", 1, 120, RRF_K)
            filter_field = st.selectbox(
                "Limit to",
                ["No filter", "Source", "Dataset"],
                help="Filter using values found in your indexed dataset.",
            )
            options = source_values if filter_field == "Source" else dataset_values
            filter_value = None
            if filter_field != "No filter":
                filter_value = st.selectbox("Value", ["All"] + options)
            compare_mode = st.checkbox("Compare all search methods", value=False)
            show_scores = st.checkbox("Show relevance scores", value=True)

        st.caption(f"{len(source_values)} source label(s) · Ready for search")

    st.markdown(
        '<div class="eyebrow">YOUR KNOWLEDGE WORKSPACE</div>'
        '<div class="hero-title">Find a clear answer.</div>'
        '<div class="hero-copy">Search your technical library and explore the passages behind each match.</div>',
        unsafe_allow_html=True,
    )

    if active_page == "History":
        st.subheader("Recent searches")
        history = st.session_state.search_history
        if not history:
            st.info("Your recent searches will appear here after you search.")
        for index, item in enumerate(history):
            col_query, col_action = st.columns([5, 1])
            col_query.markdown(f"**{html.escape(item)}**")
            col_action.button(
                "Search again",
                key=f"history_search_{index}",
                on_click=open_search,
                args=(item,),
                width="stretch",
            )
        return

    if active_page == "Saved":
        st.subheader("Saved searches")
        saved = load_saved_searches()
        st.session_state.saved_searches = saved
        if not saved:
            st.info("Save a search from its results to keep it here for next time.")
        for index, item in enumerate(saved):
            col_query, col_search, col_remove = st.columns([4, 1, 1])
            col_query.markdown(f"**{html.escape(item)}**")
            col_search.button(
                "Open",
                key=f"saved_open_{index}",
                on_click=open_search,
                args=(item,),
                width="stretch",
            )
            col_remove.button(
                "Remove",
                key=f"saved_remove_{index}",
                on_click=remove_saved_search,
                args=(item,),
                width="stretch",
            )
        return

    if not all_ready:
        st.warning("Search indexes are missing or out of date. Follow the specific instruction below, then restart the app.")
        for item in missing:
            st.caption(item)
        return

    with st.form("knowledge_search", clear_on_submit=False):
        input_col, submit_col = st.columns([6, 1])
        with input_col:
            query = st.text_input(
                "Ask your library",
                placeholder="Try: How do NumPy arrays use shape and broadcasting?",
                label_visibility="collapsed",
                key="query_input",
            )
        with submit_col:
            search_clicked = st.form_submit_button("Search  →", type="primary", width="stretch")

    run_loaded_query = st.session_state.pop("run_loaded_query", False)
    should_search = search_clicked or run_loaded_query
    metadata_filter = None
    if filter_field == "Source" and filter_value and filter_value != "All":
        metadata_filter = {"source": filter_value}
    elif filter_field == "Dataset" and filter_value and filter_value != "All":
        metadata_filter = {"dataset": filter_value}
    search_settings = {
        "method": retrieval_method,
        "top_k": final_top_k,
        "dense_top_k": dense_top_k,
        "bm25_top_k": bm25_top_k,
        "rrf_k": rrf_k,
        "filter": metadata_filter,
        "compare": compare_mode,
    }

    if should_search:
        if not query.strip():
            st.info("Enter a question or a few keywords to search your library.")
            return
        if not all_ready:
            st.warning("Search indexes are missing. Build them before searching.")
            return

        try:
            from src.dense_search import DenseRetriever
            from src.hybrid_search import HybridRetriever

            embedding_model = load_embedding_model()
            vector_store = load_vector_store()
            bm25_retriever = load_bm25_retriever()
            dense_retriever = DenseRetriever(vector_store=vector_store, embedding_model=embedding_model)
            hybrid_retriever = HybridRetriever(
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever,
                vector_store=vector_store,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                final_top_k=final_top_k,
                rrf_k=rrf_k,
            )

            comparison = {}
            started = time.perf_counter()
            with st.spinner("Searching your library…"):
                if compare_mode or retrieval_method == "Dense Search":
                    t0 = time.perf_counter()
                    comparison["Dense Search"] = (
                        dense_retriever.search(query, top_k=final_top_k, metadata_filter=metadata_filter),
                        (time.perf_counter() - t0) * 1000,
                    )
                if compare_mode or retrieval_method == "BM25 Keyword":
                    t0 = time.perf_counter()
                    comparison["BM25 Keyword"] = (
                        bm25_retriever.search(query, top_k=final_top_k, metadata_filter=metadata_filter),
                        (time.perf_counter() - t0) * 1000,
                    )
                if compare_mode or retrieval_method == "Hybrid RRF":
                    response = hybrid_retriever.search(query, metadata_filter=metadata_filter)
                    comparison["Hybrid RRF"] = (response["results"], response["latency"].get("total_ms", 0.0))

            primary_method = "Hybrid RRF" if "Hybrid RRF" in comparison else retrieval_method
            results, elapsed_ms = comparison[primary_method]
            if compare_mode:
                elapsed_ms = (time.perf_counter() - started) * 1000
            st.session_state.last_query = query.strip()
            st.session_state.last_results = results
            st.session_state.last_method = primary_method
            st.session_state.last_elapsed_ms = elapsed_ms
            st.session_state.last_comparison = comparison
            st.session_state.last_settings = search_settings
            history = [query.strip()] + [item for item in st.session_state.search_history if item.casefold() != query.strip().casefold()]
            st.session_state.search_history = history[:30]
        except Exception as exc:
            st.error(f"Search could not complete: {exc}")
            return

    last_query = st.session_state.get("last_query", "")
    if not last_query or query.strip() != last_query:
        st.info("Enter a question above to search. Your matching passages will appear here with source details.")
        return
    if st.session_state.get("last_settings") != search_settings:
        st.info("Your search options changed. Click **Search** again to refresh these results.")
        return

    results = st.session_state.get("last_results", [])
    last_method = st.session_state.get("last_method", retrieval_method)
    elapsed_ms = st.session_state.get("last_elapsed_ms", 0.0)
    saved = load_saved_searches()
    already_saved = last_query.casefold() in {item.casefold() for item in saved}
    save_col, info_col = st.columns([1, 5])
    save_col.button(
        "✓ Saved" if already_saved else "☆ Save search",
        key="save_current_search",
        disabled=already_saved,
        on_click=None if already_saved else save_search,
        args=() if already_saved else (last_query,),
    )
    info_col.caption("Saved searches stay available after you close the app.")

    if not results:
        st.warning("No matching passages found. Try a shorter query or change the advanced source filter.")
        return

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Matches", len(results))
    metric_b.metric("Search method", last_method)
    metric_c.metric("Search time", f"{elapsed_ms:.1f} ms")

    answer_parts = compose_grounded_answer(last_query, results)
    source_names = list(dict.fromkeys(
        str(part["metadata"].get("library") or part["metadata"].get("dataset") or part["metadata"].get("source") or "Library")
        for part in answer_parts if part["supported"]
    ))
    source_label = ", ".join(source_names) if source_names else "No matching source"
    answer_lines = []
    for part in answer_parts:
        label = str(part["label"])
        text = str(part["text"])
        text = re.sub(rf"^{re.escape(label)}\s*:\s*", "", text, count=1, flags=re.IGNORECASE)
        if label.casefold() == "answer" or text.casefold().startswith(label.casefold()):
            answer_lines.append(f'<div class="answer-part">{html.escape(text)}</div>')
        else:
            answer_lines.append(
                f'<div class="answer-part"><strong>{html.escape(label)}:</strong> '
                f'{html.escape(text)}</div>'
            )
    answer_html = "".join(answer_lines)
    st.markdown(
        f"""<div class="answer-card">
        <div class="answer-label"><span>✦ &nbsp; Answer from your library</span><span class="pill">{html.escape(source_label)}</span></div>
        <div class="answer-text">{answer_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("Each part is matched to a supporting passage below; unsupported parts are called out instead of guessed.")

    st.subheader("Supporting passages")
    comparisons = st.session_state.get("last_comparison", {})
    if len(comparisons) > 1:
        tabs = st.tabs(list(comparisons))
        for tab, (method, (method_results, method_time)) in zip(tabs, comparisons.items()):
            with tab:
                st.caption(f"{len(method_results)} matches · {method_time:.1f} ms")
                for index, result in enumerate(method_results, 1):
                    render_source_card(result, index, show_scores)
    else:
        for index, result in enumerate(results[1:], 1):
            render_source_card(result, index, show_scores)
        if len(results) == 1:
            st.caption("This is the only matching passage in the current result set.")


if __name__ == "__main__":
    main()
