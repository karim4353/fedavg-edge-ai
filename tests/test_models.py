"""
test_models.py - Tests for MLP model implementations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.models import (
    MLPModel, create_full_model, create_edge_model,
    relu, softmax, one_hot, cross_entropy_loss
)


class TestActivations:
    """Tests for activation functions."""

    def test_relu_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_relu_negative(self):
        x = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(relu(x), np.zeros(3))

    def test_relu_mixed(self):
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(relu(x), np.array([0.0, 0.0, 1.0]))

    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        np.testing.assert_almost_equal(np.sum(result), 1.0)

    def test_softmax_all_positive(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        assert np.all(result > 0)

    def test_one_hot_shape(self):
        y = np.array([0, 1, 2, 3])
        oh = one_hot(y, 10)
        assert oh.shape == (4, 10)

    def test_one_hot_values(self):
        y = np.array([0, 5, 9])
        oh = one_hot(y, 10)
        assert oh[0, 0] == 1
        assert oh[1, 5] == 1
        assert oh[2, 9] == 1
        assert np.sum(oh) == 3


class TestMLPModel:
    """Tests for the MLP model."""

    def test_forward_output_shape(self):
        model = create_full_model(seed=42)
        X = np.random.randn(5, 64)
        output = model.forward(X)
        assert output.shape == (5, 10)

    def test_forward_probabilities_sum_to_one(self):
        model = create_full_model(seed=42)
        X = np.random.randn(10, 64)
        output = model.forward(X)
        row_sums = np.sum(output, axis=1)
        np.testing.assert_almost_equal(row_sums, np.ones(10))

    def test_predict_returns_class_labels(self):
        model = create_full_model(seed=42)
        X = np.random.randn(10, 64)
        predictions = model.predict(X)
        assert predictions.shape == (10,)
        assert all(0 <= p <= 9 for p in predictions)

    def test_deterministic_forward(self):
        model = create_full_model(seed=42)
        X = np.random.randn(5, 64)
        out1 = model.forward(X).copy()
        out2 = model.forward(X).copy()
        np.testing.assert_array_equal(out1, out2)

    def test_get_set_params(self):
        model = create_full_model(seed=42)
        params = model.get_params()

        model2 = create_full_model(seed=99)  # different seed
        model2.set_params(params)

        X = np.random.randn(5, 64)
        np.testing.assert_array_almost_equal(
            model.forward(X), model2.forward(X)
        )

    def test_count_params(self):
        full = create_full_model(seed=42)
        edge = create_edge_model(seed=42)
        assert full.count_params() > edge.count_params()
        assert full.count_params() > 0
        assert edge.count_params() > 0

    def test_full_model_architecture(self):
        model = create_full_model(seed=42)
        assert model.input_dim == 64
        assert model.hidden_dims == (128, 64)
        assert model.output_dim == 10
        assert len(model.weights) == 3  # 3 layers

    def test_edge_model_architecture(self):
        model = create_edge_model(seed=42)
        assert model.input_dim == 64
        assert model.hidden_dims == (48, 24)
        assert model.output_dim == 10
        assert len(model.weights) == 3

    def test_training_reduces_loss(self):
        """Training should reduce the loss on the training data."""
        model = create_full_model(seed=42)
        X = np.random.RandomState(42).randn(100, 64)
        y = np.random.RandomState(42).randint(0, 10, 100)

        loss_before = model.evaluate(X, y)['loss']

        # Train for a few steps
        y_oh = one_hot(y, 10)
        for _ in range(20):
            model.forward(X)
            model.backward(y_oh, 0.1)

        loss_after = model.evaluate(X, y)['loss']
        assert loss_after < loss_before


class TestModelSize:
    """Tests for model size estimation."""

    def test_model_size_bytes(self):
        model = create_full_model(seed=42)
        size64 = model.get_model_size_bytes('float64')
        size32 = model.get_model_size_bytes('float32')
        size8 = model.get_model_size_bytes('int8')

        assert size64 == size32 * 2
        assert size64 == size8 * 8
        assert size64 > 0
