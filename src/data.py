"""
data.py - Data loading, IID and non-IID partitioning for federated learning.

Reproduces the partitioning strategies from McMahan et al. (2017):
  - IID: shuffle data uniformly and distribute evenly across clients
  - Non-IID: sort by label, create shards, assign 2 shards per client
    (pathological non-IID as described in the paper, Section 3)

We use scikit-learn's digits dataset (10 classes, 1797 samples, 64 features)
as a lightweight proxy for MNIST. This is scientifically reasonable because:
  - Same task structure (10-class digit classification)
  - Small enough for CI/CD and laptop execution
  - Deterministic and built into scikit-learn (no downloads needed)
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_digit_data(seed=42):
    """Load and preprocess the digits dataset.

    Returns:
        X_train, X_test, y_train, y_test: numpy arrays
        The features are standardized to zero mean and unit variance.
    """
    digits = load_digits()
    X, y = digits.data, digits.target

    # Normalize to [0, 1] then standardize
    X = X / 16.0  # digits pixel values are 0-16

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def partition_iid(X, y, num_clients, seed=42):
    """IID partitioning: shuffle and distribute evenly.

    As described in McMahan et al. Section 3:
    'the data is shuffled, and then partitioned into [K] clients
     each receiving [n/K] examples'

    Args:
        X: feature matrix (n_samples, n_features)
        y: label vector (n_samples,)
        num_clients: number of clients K
        seed: random seed for reproducibility

    Returns:
        list of (X_k, y_k) tuples, one per client
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)

    # Split into num_clients roughly equal parts
    splits = np.array_split(indices, num_clients)

    client_data = []
    for split in splits:
        client_data.append((X[split], y[split]))

    return client_data


def partition_non_iid(X, y, num_clients, shards_per_client=2, seed=42):
    """Non-IID partitioning (pathological).

    As described in McMahan et al. Section 3:
    'we first sort the data by digit label, divide it into 200 shards
     of size 300, and assign each of 100 clients 2 shards'

    This means most clients only see examples from 2 digit classes,
    creating a highly heterogeneous data distribution.

    Args:
        X: feature matrix
        y: label vector
        num_clients: number of clients K
        shards_per_client: number of shards assigned to each client (default=2)
        seed: random seed

    Returns:
        list of (X_k, y_k) tuples, one per client
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    num_shards = num_clients * shards_per_client

    # Sort by label
    sorted_indices = np.argsort(y, kind='stable')

    # Divide into shards
    shard_size = n // num_shards
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size
        if i == num_shards - 1:
            end = n  # last shard gets remaining
        shards.append(sorted_indices[start:end])

    # Shuffle shards and assign to clients
    shard_indices = list(range(num_shards))
    rng.shuffle(shard_indices)

    client_data = []
    for i in range(num_clients):
        client_shards = shard_indices[i * shards_per_client:(i + 1) * shards_per_client]
        indices = np.concatenate([shards[s] for s in client_shards])
        client_data.append((X[indices], y[indices]))

    return client_data


def get_client_stats(client_data):
    """Compute statistics about client data distribution.

    Args:
        client_data: list of (X_k, y_k) tuples

    Returns:
        list of dicts with 'num_samples' and 'label_distribution'
    """
    stats = []
    for X_k, y_k in client_data:
        unique, counts = np.unique(y_k, return_counts=True)
        stats.append({
            'num_samples': len(y_k),
            'label_distribution': dict(zip(unique.tolist(), counts.tolist())),
            'num_classes': len(unique),
        })
    return stats
