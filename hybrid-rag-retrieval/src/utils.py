"""
utils.py - Shared helper utilities for the Hybrid RAG system.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

# ── Logging setup ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ── File helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSON-Lines file and return a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Save a list of dicts to a JSON-Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> Any:
    """Load a single JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ── Timing helpers ────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer that records elapsed seconds."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def measure_latency(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) and return (result, elapsed_seconds).
    Useful for one-off latency measurements.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ── Percentile helpers ────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    """
    Compute the p-th percentile (0–100) of a list of floats.
    Uses linear interpolation (same as numpy's default).
    """
    import numpy as np
    return float(np.percentile(values, p))
