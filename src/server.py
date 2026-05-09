"""Server-side aggregation primitives for federated learning."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def weighted_average(
    params_list: Sequence[Sequence[np.ndarray]],
    weights: Sequence[float],
) -> List[np.ndarray]:
    """Per-tensor weighted average of multiple client parameter sets.

    Implements the FedAvg server step:

        w_{t+1} = sum_k (n_k / n) * w_k

    where `weights[k] = n_k`. This works for any pytree-style flat list of
    tensors, as long as all clients submit the same shapes.
    """
    if len(params_list) == 0:
        raise ValueError("params_list is empty")
    if len(params_list) != len(weights):
        raise ValueError("params_list and weights must have the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("sum of weights must be positive")

    out: List[np.ndarray] = []
    for tensors in zip(*params_list):
        # tensors: tuple of np.ndarrays, one per client, all same shape
        avg = np.zeros_like(tensors[0], dtype=np.float32)
        for t, w in zip(tensors, weights):
            avg = avg + (float(w) / total) * t.astype(np.float32)
        out.append(avg.astype(np.float32))
    return out
