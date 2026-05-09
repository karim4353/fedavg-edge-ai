"""
test_data.py - Tests for data loading and partitioning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.data import load_digit_data, partition_iid, partition_non_iid, get_client_stats


class TestDataLoading:
    """Tests for data loading."""

    def test_load_returns_correct_shapes(self):
        """Data should have correct dimensions."""
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        assert X_train.shape[1] == 64  # 8x8 pixels
        assert X_test.shape[1] == 64
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_load_is_deterministic(self):
        """Same seed should give same data."""
        X1, _, y1, _ = load_digit_data(seed=42)
        X2, _, y2, _ = load_digit_data(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_all_classes_present(self):
        """All 10 digits should be in training data."""
        _, _, y_train, _ = load_digit_data(seed=42)
        assert len(np.unique(y_train)) == 10

    def test_train_test_split_ratio(self):
        """Default split should be 80/20."""
        X_train, X_test, _, _ = load_digit_data(seed=42)
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05


class TestPartitionIID:
    """Tests for IID partitioning."""

    def test_partition_preserves_all_data(self):
        """All data points should appear exactly once across clients."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        clients = partition_iid(X_train, y_train, 10, seed=42)

        total_samples = sum(len(X) for X, _ in clients)
        assert total_samples == len(X_train)

    def test_partition_creates_correct_num_clients(self):
        """Should create the requested number of client partitions."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        for n_clients in [3, 5, 10, 20]:
            clients = partition_iid(X_train, y_train, n_clients, seed=42)
            assert len(clients) == n_clients

    def test_partition_roughly_balanced(self):
        """Client datasets should be roughly equal in size."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        clients = partition_iid(X_train, y_train, 10, seed=42)
        sizes = [len(X) for X, _ in clients]
        assert max(sizes) - min(sizes) <= 2  # at most 2 difference

    def test_iid_has_multiple_classes(self):
        """IID clients should have examples from multiple classes."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        clients = partition_iid(X_train, y_train, 10, seed=42)
        stats = get_client_stats(clients)
        for s in stats:
            assert s['num_classes'] >= 5  # IID should have most classes

    def test_partition_is_deterministic(self):
        """Same seed should produce same partition."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        c1 = partition_iid(X_train, y_train, 10, seed=42)
        c2 = partition_iid(X_train, y_train, 10, seed=42)
        for (X1, y1), (X2, y2) in zip(c1, c2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)


class TestPartitionNonIID:
    """Tests for non-IID (pathological) partitioning."""

    def test_partition_preserves_all_data(self):
        """All data points should appear across clients."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        clients = partition_non_iid(X_train, y_train, 10, seed=42)
        total_samples = sum(len(X) for X, _ in clients)
        # Might lose a few due to shard rounding, but should be close
        assert total_samples >= len(X_train) * 0.9

    def test_non_iid_has_fewer_classes(self):
        """Non-IID clients should have fewer classes than IID."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        clients_iid = partition_iid(X_train, y_train, 10, seed=42)
        clients_noniid = partition_non_iid(X_train, y_train, 10, seed=42)

        avg_iid = np.mean([s['num_classes'] for s in get_client_stats(clients_iid)])
        avg_noniid = np.mean([s['num_classes'] for s in get_client_stats(clients_noniid)])

        assert avg_noniid < avg_iid

    def test_partition_creates_correct_num_clients(self):
        """Should create the requested number of client partitions."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        clients = partition_non_iid(X_train, y_train, 10, seed=42)
        assert len(clients) == 10

    def test_partition_is_deterministic(self):
        """Same seed should produce same partition."""
        X_train, _, y_train, _ = load_digit_data(seed=42)
        c1 = partition_non_iid(X_train, y_train, 10, seed=42)
        c2 = partition_non_iid(X_train, y_train, 10, seed=42)
        for (X1, y1), (X2, y2) in zip(c1, c2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)
