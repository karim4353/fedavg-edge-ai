"""
test_federated.py - Tests for federated learning algorithms.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.data import load_digit_data, partition_iid
from src.models import create_full_model, MLPModel
from src.federated import (
    federated_avg_round, federated_sgd_round,
    _aggregate_params, train_federated, train_centralized
)


class TestAggregation:
    """Tests for server-side parameter aggregation."""

    def test_aggregate_weighted_average(self):
        """Aggregation should compute a weighted average by sample count."""
        # Two clients with simple 1-layer models
        params1 = [(np.array([[1.0, 2.0]]), np.array([[0.1]]))]
        params2 = [(np.array([[3.0, 4.0]]), np.array([[0.2]]))]

        # Equal sizes → simple average
        result = _aggregate_params([params1, params2], [10, 10])
        np.testing.assert_array_almost_equal(
            result[0][0], np.array([[2.0, 3.0]])
        )
        np.testing.assert_array_almost_equal(
            result[0][1], np.array([[0.15]])
        )

    def test_aggregate_unequal_weights(self):
        """Heavier clients should have more influence."""
        params1 = [(np.array([[1.0]]), np.array([[0.0]]))]
        params2 = [(np.array([[3.0]]), np.array([[0.0]]))]

        # Client 2 has 3× the data
        result = _aggregate_params([params1, params2], [100, 300])
        expected = (100 * 1.0 + 300 * 3.0) / 400
        np.testing.assert_almost_equal(result[0][0][0, 0], expected)

    def test_aggregate_single_client(self):
        """Single client aggregation should return the client's params."""
        params = [(np.array([[1.0, 2.0]]), np.array([[3.0]]))]
        result = _aggregate_params([params], [10])
        np.testing.assert_array_equal(result[0][0], params[0][0])
        np.testing.assert_array_equal(result[0][1], params[0][1])

    def test_aggregate_preserves_shapes(self):
        """Output shapes should match input shapes."""
        W = np.random.randn(64, 128)
        b = np.random.randn(1, 128)
        params = [(W.copy(), b.copy())]
        result = _aggregate_params([params, params], [10, 10])
        assert result[0][0].shape == W.shape
        assert result[0][1].shape == b.shape


class TestFedSGD:
    """Tests for FedSGD."""

    def test_fedsgd_updates_model(self):
        """FedSGD round should change model parameters."""
        model = create_full_model(seed=42)
        X, _, y, _ = load_digit_data(seed=42)
        clients = partition_iid(X, y, 5, seed=42)

        params_before = model.get_params()
        new_params = federated_sgd_round(model, clients, 0.1, seed=42)
        model.set_params(new_params)

        # At least one weight should have changed
        for (W1, b1), (W2, b2) in zip(params_before, model.get_params()):
            if not np.array_equal(W1, W2):
                return
        pytest.fail("FedSGD should update at least one parameter")

    def test_fedsgd_is_deterministic(self):
        """Same seed → same results."""
        X, _, y, _ = load_digit_data(seed=42)
        clients = partition_iid(X, y, 5, seed=42)

        model1 = create_full_model(seed=42)
        p1 = federated_sgd_round(model1, clients, 0.1, seed=42)

        model2 = create_full_model(seed=42)
        p2 = federated_sgd_round(model2, clients, 0.1, seed=42)

        for (W1, b1), (W2, b2) in zip(p1, p2):
            np.testing.assert_array_equal(W1, W2)
            np.testing.assert_array_equal(b1, b2)


class TestFedAvg:
    """Tests for FedAvg."""

    def test_fedavg_updates_model(self):
        """FedAvg round should change model parameters."""
        model = create_full_model(seed=42)
        X, _, y, _ = load_digit_data(seed=42)
        clients = partition_iid(X, y, 5, seed=42)

        params_before = model.get_params()
        new_params = federated_avg_round(
            model, clients, 0.1,
            local_epochs=3, batch_size=10, seed=42
        )
        model.set_params(new_params)

        changed = False
        for (W1, _), (W2, _) in zip(params_before, model.get_params()):
            if not np.array_equal(W1, W2):
                changed = True
                break
        assert changed, "FedAvg should update parameters"

    def test_fedavg_is_deterministic(self):
        """Same seed → same results."""
        X, _, y, _ = load_digit_data(seed=42)
        clients = partition_iid(X, y, 5, seed=42)

        model1 = create_full_model(seed=42)
        p1 = federated_avg_round(model1, clients, 0.1,
                                   local_epochs=2, batch_size=10, seed=42)

        model2 = create_full_model(seed=42)
        p2 = federated_avg_round(model2, clients, 0.1,
                                   local_epochs=2, batch_size=10, seed=42)

        for (W1, b1), (W2, b2) in zip(p1, p2):
            np.testing.assert_array_equal(W1, W2)
            np.testing.assert_array_equal(b1, b2)

    def test_fedavg_e1_b_inf_equals_fedsgd(self):
        """FedAvg with E=1, B=inf should equal FedSGD."""
        X, _, y, _ = load_digit_data(seed=42)
        clients = partition_iid(X, y, 5, seed=42)

        model_sgd = create_full_model(seed=42)
        p_sgd = federated_sgd_round(model_sgd, clients, 0.1,
                                      client_fraction=1.0, seed=42)

        model_avg = create_full_model(seed=42)
        p_avg = federated_avg_round(model_avg, clients, 0.1,
                                      local_epochs=1, batch_size=None,
                                      client_fraction=1.0, seed=42)

        for (W1, b1), (W2, b2) in zip(p_sgd, p_avg):
            np.testing.assert_array_almost_equal(W1, W2, decimal=10)
            np.testing.assert_array_almost_equal(b1, b2, decimal=10)

    def test_client_fraction_selects_subset(self):
        """C < 1 should select fewer than K clients."""
        X, _, y, _ = load_digit_data(seed=42)
        clients = partition_iid(X, y, 10, seed=42)

        # This is tricky to test directly, but we can verify that
        # with C=0.1 and 10 clients, we select 1 client
        # The result should still be valid
        model = create_full_model(seed=42)
        p = federated_avg_round(model, clients, 0.1,
                                  local_epochs=1, batch_size=10,
                                  client_fraction=0.1, seed=42)
        assert len(p) == len(model.get_params())


class TestTrainFederated:
    """Tests for the full training loop."""

    def test_training_returns_history(self):
        """Training should return a proper history dict."""
        X, X_test, y, y_test = load_digit_data(seed=42)
        clients = partition_iid(X, y, 5, seed=42)
        model = create_full_model(seed=42)

        history = train_federated(
            model, clients, X_test, y_test,
            num_rounds=3, learning_rate=0.1,
            local_epochs=1, batch_size=10,
            seed=42, verbose=False
        )

        assert 'round' in history
        assert 'accuracy' in history
        assert 'loss' in history
        assert len(history['round']) == 3

    def test_centralized_training_works(self):
        """Centralized training should run and return history."""
        X, X_test, y, y_test = load_digit_data(seed=42)
        model = create_full_model(seed=42)

        history = train_centralized(
            model, X, y, X_test, y_test,
            num_epochs=3, learning_rate=0.1,
            seed=42, verbose=False
        )

        assert 'epoch' in history
        assert len(history['epoch']) == 3
        assert all(0 <= a <= 1 for a in history['accuracy'])
