"""Utility helpers: deterministic seeding, IO."""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python's `random` and NumPy for fully deterministic experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def stopwatch() -> Iterator[dict]:
    """Context manager yielding a dict with `elapsed` (seconds) when exiting."""
    state: dict = {"elapsed": 0.0}
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - t0
