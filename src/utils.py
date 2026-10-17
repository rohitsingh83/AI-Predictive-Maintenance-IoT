"""
utils.py
========
Small shared helpers: logging, a timer context-manager, a console banner,
safe JSON dump, and a couple of reusable plotting/formatting utilities.

Keeping these here means every other module stays short and readable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "pm", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints nicely to the console."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def banner(title: str, width: int = 78, char: str = "=") -> None:
    """Print an eye-catching section banner (nice for training logs/screenshots)."""
    print("\n" + char * width)
    print(f"  {title.upper()}")
    print(char * width)


@contextmanager
def timer(label: str, logger: logging.Logger | None = None):
    """with timer('training', log): ... -> logs how long the block took."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        (logger.info if logger else print)(f"[{label}] finished in {dt:.2f}s")


# ---------------------------------------------------------------------------
# JSON / numeric helpers
# ---------------------------------------------------------------------------
def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas types into JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)
    get_logger().info("Saved %s", path)


def risk_from_probability(prob: float, bands) -> tuple[str, str]:
    """Map a failure probability to (severity, recommended action)."""
    for threshold, severity, action in bands:
        if prob >= threshold:
            return severity, action
    return "LOW", "Healthy - continue normal operation"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def print_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n--- {title} ---")
    print(df.to_string(index=False))
