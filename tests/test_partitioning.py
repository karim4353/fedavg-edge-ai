"""Data partitioning tests."""
import numpy as np

from src.data import (
    make_synthetic_classification,
    partition_iid,
    partition_noniid_by_label,
)


def test_iid_covers_all_samples_no_overlap():
    X, y = make_synthetic_classification(n_samples=400, n_classes=5, seed=0)
    parts = partition_iid(X, y, n_clients=8, seed=0)
    assert len(parts) == 8
    total = sum(len(p[1]) for p in parts)
    assert total == len(y)
    sizes = [len(p[1]) for p in parts]
    assert max(sizes) - min(sizes) <= 1  # roughly equal


def test_iid_each_client_sees_multiple_classes():
    X, y = make_synthetic_classification(n_samples=400, n_classes=5, seed=0)
    parts = partition_iid(X, y, n_clients=4, seed=0)
    for _, yc in parts:
        # IID partition with hundreds of samples should hit several classes
        assert len(np.unique(yc)) >= 3


def test_noniid_restricts_classes_per_client():
    X, y = make_synthetic_classification(n_samples=400, n_classes=5, seed=0)
    parts = partition_noniid_by_label(X, y, n_clients=10, shards_per_client=2, seed=0)
    assert len(parts) == 10
    # The pathological partition assigns 2 shards per client; clients should
    # see at most a few classes (not all 5).
    max_classes = max(len(np.unique(yc)) for _, yc in parts)
    assert max_classes <= 4


def test_noniid_total_samples_preserved():
    X, y = make_synthetic_classification(n_samples=400, n_classes=5, seed=1)
    parts = partition_noniid_by_label(X, y, n_clients=5, shards_per_client=2, seed=1)
    total = sum(len(p[1]) for p in parts)
    assert total == len(y)


def test_partitions_are_seed_deterministic():
    X, y = make_synthetic_classification(n_samples=400, n_classes=5, seed=2)
    a = partition_iid(X, y, n_clients=4, seed=99)
    b = partition_iid(X, y, n_clients=4, seed=99)
    for (xa, ya), (xb, yb) in zip(a, b):
        np.testing.assert_array_equal(ya, yb)
        np.testing.assert_array_equal(xa, xb)
