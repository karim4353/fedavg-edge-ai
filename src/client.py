"""Client-side local training routines."""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

from .model import MLP


def _iter_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = len(y)
    idx = rng.permutation(n)
    bs = max(1, batch_size)
    for start in range(0, n, bs):
        b = idx[start : start + bs]
        yield X[b], y[b]


def local_train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int = 0,
) -> List[np.ndarray]:
    """Run `epochs` local SGD passes; mutate `model` in place; return its params.

    `batch_size` may be larger than the local dataset, in which case each
    epoch performs a single full-batch step (this is the FedSGD regime when
    `epochs == 1`).
    """
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for xb, yb in _iter_minibatches(X, y, batch_size, rng):
            _, cache = model.forward(xb)
            grads = model.backward(cache, yb)
            model.step(grads, lr)
    return model.get_params()
