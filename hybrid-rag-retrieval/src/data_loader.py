"""
data_loader.py - Download and load the MS MARCO passage dataset.

MS MARCO is a large-scale reading-comprehension / passage-retrieval dataset
from Microsoft.  We use the HuggingFace 'microsoft/ms_marco' dataset.

Usage
-----
    from src.data_loader import load_ms_marco
    passages = load_ms_marco(size=10_000, mode="demo")
"""

from pathlib import Path
from typing import Optional

from src.config import (
    DEMO_SIZE,
    FULL_DATASET_SIZE,
    PASSAGES_FILE,
    RAW_DATA_DIR,
)
from src.utils import get_logger, load_jsonl

logger = get_logger(__name__)


def load_ms_marco(
    size: Optional[int] = None,
    mode: str = "demo",
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """
    Download (or load from cache) the MS MARCO passage dataset and return
    a list of passage dicts.

    Parameters
    ----------
    size  : int, optional
        Number of passages to return.  Defaults to DEMO_SIZE in demo mode
        and FULL_DATASET_SIZE in full mode.
    mode  : str
        'demo'  – fast, loads a small subset (good for development)
        'full'  – loads the full dataset (may take a while)
    cache_dir : str, optional
        Directory to cache the HuggingFace dataset files.

    Returns
    -------
    List of dicts, each with keys:
        id, text, metadata (source, dataset, query_id, is_selected)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required.  "
            "Install it with:  pip install datasets"
        )

    if size is None:
        size = DEMO_SIZE if mode == "demo" else FULL_DATASET_SIZE

    cache_path = Path(cache_dir) if cache_dir else RAW_DATA_DIR
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading MS MARCO dataset (size={size}, mode={mode}) …")

    # Try multiple dataset paths in order of preference
    # NOTE: The correct HuggingFace path is "microsoft/ms_marco" (with namespace)
    dataset_configs = [
        ("microsoft/ms_marco", "v2.1", "train"),
        ("microsoft/ms_marco", "v1.1", "train"),
    ]

    dataset = None
    last_error = None

    for dataset_name, config_name, split in dataset_configs:
        try:
            logger.info(f"Trying {dataset_name} ({config_name}) …")
            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=True,
                cache_dir=str(cache_path),
            )
            logger.info(f"Successfully loaded {dataset_name} ({config_name})")
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Could not load {dataset_name} ({config_name}): {e}")
            continue

    if dataset is None:
        raise RuntimeError(
            f"Failed to load MS MARCO dataset. Last error: {last_error}\n"
            "Make sure you have an internet connection and the 'datasets' "
            "package installed (pip install datasets)."
        )

    passages: list[dict] = []
    seen_texts: set[str] = set()
    passage_id = 0

    logger.info("Extracting passages from dataset …")

    for example in dataset:
        # Each MS MARCO example has a list of passages and a query
        query_id   = str(example.get("query_id", ""))
        query_text = example.get("query", "")
        passages_data = example.get("passages", {})

        passage_texts     = passages_data.get("passage_text", [])
        is_selected_flags = passages_data.get("is_selected", [])
        urls              = passages_data.get("url", [])

        for i, text in enumerate(passage_texts):
            if not text or not text.strip():
                continue
            text = text.strip()

            # Skip exact duplicates
            if text in seen_texts:
                continue
            seen_texts.add(text)

            is_selected = (
                bool(is_selected_flags[i])
                if i < len(is_selected_flags)
                else False
            )
            url = urls[i] if i < len(urls) else ""

            passage = {
                "id": f"passage_{passage_id}",
                "text": text,
                "metadata": {
                    "source":      "ms_marco",
                    "dataset":     "ms_marco",
                    "query_id":    query_id,
                    "query_text":  query_text,
                    "is_selected": is_selected,
                    "url":         url,
                    "passage_idx": i,
                },
            }
            passages.append(passage)
            passage_id += 1

            if len(passages) >= size:
                break

        if len(passages) >= size:
            break

    logger.info(f"Loaded {len(passages)} passages from MS MARCO.")
    return passages


def load_ms_marco_queries(
    size: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """
    Load a sample of MS MARCO queries together with their relevant passage texts.
    Used for evaluation (computing Recall, Precision, MRR).

    Returns
    -------
    List of dicts:
        query_id, query_text, relevant_passage_texts (list of str)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    size = size or 500
    cache_path = RAW_DATA_DIR
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading MS MARCO queries for evaluation (size={size}) …")

    dataset_configs = [
        ("microsoft/ms_marco", "v2.1", "validation"),
        ("microsoft/ms_marco", "v1.1", "validation"),
        ("microsoft/ms_marco", "v2.1", "train"),
    ]

    dataset = None
    for dataset_name, config_name, split in dataset_configs:
        try:
            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                streaming=True,
                cache_dir=str(cache_path),
            )
            logger.info(f"Loaded {dataset_name} ({config_name}, {split}) for queries")
            break
        except Exception as e:
            logger.warning(f"Could not load {dataset_name} ({config_name}, {split}): {e}")
            continue

    if dataset is None:
        logger.error("Could not load any MS MARCO query dataset.")
        return []

    queries = []
    for example in dataset:
        query_id   = str(example.get("query_id", ""))
        query_text = example.get("query", "").strip()
        if not query_text:
            continue

        passages_data     = example.get("passages", {})
        passage_texts     = passages_data.get("passage_text", [])
        is_selected_flags = passages_data.get("is_selected", [])

        relevant_texts = [
            t.strip()
            for t, sel in zip(passage_texts, is_selected_flags)
            if sel and t.strip()
        ]

        if not relevant_texts:
            continue

        queries.append(
            {
                "query_id":              query_id,
                "query_text":            query_text,
                "relevant_passage_texts": relevant_texts,
            }
        )
        if len(queries) >= size:
            break

    logger.info(f"Loaded {len(queries)} evaluation queries.")
    return queries


def load_indexed_ms_marco_queries(
    size: Optional[int] = None,
    passages_path: Optional[str | Path] = None,
) -> list[dict]:
    """Load evaluation queries whose labeled passages are in the local index.

    The demo index is built from MS MARCO's training split while the hosted
    validation split contains different passage pools.  Evaluating the former
    against the latter therefore measures corpus mismatch rather than retrieval
    quality.  This helper builds an in-corpus evaluation set from the original
    query text and ``is_selected`` labels persisted with indexed passages.
    """
    path = Path(passages_path) if passages_path else PASSAGES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Processed passages not found at {path}. "
            "Run: python scripts/download_data.py --mode demo"
        )

    grouped: dict[str, dict] = {}
    for passage in load_jsonl(path):
        metadata = passage.get("metadata", {})
        query_id = str(metadata.get("query_id", ""))
        query_text = str(metadata.get("query_text", "")).strip()
        if not query_id or not query_text or not metadata.get("is_selected"):
            continue

        entry = grouped.setdefault(
            query_id,
            {
                "query_id": query_id,
                "query_text": query_text,
                "relevant_passage_texts": [],
            },
        )
        entry["relevant_passage_texts"].append(passage["text"])

    queries = list(grouped.values())
    if size is not None:
        queries = queries[:size]
    logger.info(f"Loaded {len(queries)} in-corpus evaluation queries.")
    return queries
