"""
Data generation and federated partitioning.

We generate a synthetic multi-class classification dataset designed to mimic
the qualitative properties of the ECG heartbeat dataset used in the team's
presentation: 5 classes, strong class imbalance (~71% majority, ~2% minority),
compact feature dimensionality suited to a TinyML-style MLP.

A fully synthetic dataset is used (rather than ECG, MNIST or CIFAR) so the
repository is reproducible offline, requires no external download, and runs
quickly enough for GitHub Actions. The generator is seeded for exact
reproducibility.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Approximate per-class proportions of the MIT-BIH/ECG heartbeat dataset
# referenced in the team presentation (N, S, V, F, Q-ish). They sum to 1.0.
_ECG_LIKE_IMBALANCE = np.array([0.71, 0.16, 0.06, 0.02, 0.05], dtype=np.float64)


def make_synthetic_classification(
    n_samples: int = 2000,
    n_features: int = 32,
    n_classes: int = 5,
    class_imbalance: bool = True,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) for a synthetic ECG-like multi-class problem.

    Each class has its own Gaussian centroid in feature space; per-class noise
    scales are drawn so that classes overlap somewhat (a learnable but not
    trivial problem at the chosen sample size).
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    rng = np.random.default_rng(seed)

    if class_imbalance:
        proportions = _ECG_LIKE_IMBALANCE.copy()
        if n_classes != len(proportions):
            # Adapt vector to requested class count, keeping skewness flavour.
            proportions = np.linspace(1.0, 0.05, n_classes)
        proportions = proportions[:n_classes]
    else:
        proportions = np.full(n_classes, 1.0 / n_classes)
    proportions = proportions / proportions.sum()

    counts = np.floor(proportions * n_samples).astype(int)
    counts[0] += n_samples - counts.sum()  # absorb rounding gap into majority

    centroids = rng.normal(0.0, 1.4, size=(n_classes, n_features))

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for k, c in enumerate(counts):
        if c == 0:
            continue
        scale = np.abs(1.0 + 0.3 * rng.standard_normal(n_features)) + 0.4
        Xk = centroids[k] + rng.normal(0.0, 1.0, size=(c, n_features)) * scale
        yk = np.full(c, k, dtype=np.int64)
        X_parts.append(Xk)
        y_parts.append(yk)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle then standardize features (zero mean, unit std).
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X.astype(np.float32), y.astype(np.int64)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(test_frac * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def partition_iid(
    X: np.ndarray, y: np.ndarray, n_clients: int, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Shuffle data and split into `n_clients` roughly-equal IID shards."""
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    splits = np.array_split(idx, n_clients)
    return [(X[s], y[s]) for s in splits]


def partition_noniid_by_label(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    shards_per_client: int = 2,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pathological non-IID partition (McMahan et al. 2017, Section 3).

    Sort by label, split into `n_clients * shards_per_client` shards, assign
    each client `shards_per_client` shards. With the default `shards_per_client=2`,
    most clients see only 1-2 distinct classes.
    """
    if n_clients < 1 or shards_per_client < 1:
        raise ValueError("invalid client/shard counts")
    rng = np.random.default_rng(seed)
    n_shards = n_clients * shards_per_client
    sort_idx = np.argsort(y, kind="stable")
    shards = np.array_split(sort_idx, n_shards)
    shard_order = rng.permutation(n_shards)

    clients: list[Tuple[np.ndarray, np.ndarray]] = []
    for c in range(n_clients):
        ids = shard_order[c * shards_per_client : (c + 1) * shards_per_client]
        client_idx = np.concatenate([shards[i] for i in ids])
        clients.append((X[client_idx], y[client_idx]))
    return clients
