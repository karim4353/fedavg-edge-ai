"""
test_edge.py - Tests for Edge AI optimizations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.models import create_full_model, create_edge_model
from src.edge_optimizations import (
    quantize_params, dequantize_params, prune_params,
    measure_inference_latency, estimate_memory_usage, compare_models
)
from src.data import load_digit_data


class TestQuantization:
    """Tests for post-training quantization."""

    def test_quantize_float32(self):
        """Float32 quantization should preserve shapes."""
        model = create_full_model(seed=42)
        params = model.get_params()
        q_params, q_info = quantize_params(params, 'float32')

        for (W_q, b_q), (W, b) in zip(q_params, params):
            assert W_q.shape == W.shape
            assert b_q.shape == b.shape
            assert W_q.dtype == np.float32

    def test_quantize_int8(self):
        """Int8 quantization should produce int8 arrays."""
        model = create_full_model(seed=42)
        params = model.get_params()
        q_params, q_info = quantize_params(params, 'int8')

        for W_q, b_q in q_params:
            assert W_q.dtype == np.int8
            assert b_q.dtype == np.int8

    def test_dequantize_roundtrip_float32(self):
        """Float32 roundtrip should be very close to original."""
        model = create_full_model(seed=42)
        params = model.get_params()
        q_params, q_info = quantize_params(params, 'float32')
        deq_params = dequantize_params(q_params, q_info)

        for (W_orig, b_orig), (W_deq, b_deq) in zip(params, deq_params):
            np.testing.assert_almost_equal(W_orig, W_deq, decimal=5)
            np.testing.assert_almost_equal(b_orig, b_deq, decimal=5)

    def test_dequantize_int8_preserves_range(self):
        """Int8 dequantized values should be in the original range."""
        model = create_full_model(seed=42)
        params = model.get_params()
        q_params, q_info = quantize_params(params, 'int8')
        deq_params = dequantize_params(q_params, q_info)

        for (W_orig, _), (W_deq, _) in zip(params, deq_params):
            # Dequantized should be roughly in the same range
            assert W_deq.min() >= W_orig.min() - 0.5
            assert W_deq.max() <= W_orig.max() + 0.5

    def test_quantized_model_still_predicts(self):
        """A dequantized model should still produce valid predictions."""
        model = create_full_model(seed=42)
        X = np.random.RandomState(42).randn(10, 64)

        params = model.get_params()
        q_params, q_info = quantize_params(params, 'int8')
        deq_params = dequantize_params(q_params, q_info)
        model.set_params(deq_params)

        predictions = model.predict(X)
        assert len(predictions) == 10
        assert all(0 <= p <= 9 for p in predictions)


class TestPruning:
    """Tests for weight pruning."""

    def test_pruning_zeros_weights(self):
        """Pruned weights should contain zeros."""
        model = create_full_model(seed=42)
        params = model.get_params()
        pruned, stats = prune_params(params, sparsity=0.5)

        total_zeros = 0
        total_params = 0
        for W, b in pruned:
            total_zeros += np.count_nonzero(W == 0)
            total_params += W.size

        assert total_zeros > 0

    def test_pruning_sparsity_matches(self):
        """Actual sparsity should be close to requested sparsity."""
        model = create_full_model(seed=42)
        params = model.get_params()
        pruned, stats = prune_params(params, sparsity=0.3)

        assert abs(stats['overall_sparsity'] - 0.3) < 0.1

    def test_zero_pruning_no_change(self):
        """0% pruning should not change the model."""
        model = create_full_model(seed=42)
        params = model.get_params()
        pruned, stats = prune_params(params, sparsity=0.0)

        for (W_orig, b_orig), (W_pruned, b_pruned) in zip(params, pruned):
            np.testing.assert_array_equal(W_orig, W_pruned)
            np.testing.assert_array_equal(b_orig, b_pruned)

    def test_pruning_preserves_shapes(self):
        """Pruning should not change weight shapes."""
        model = create_full_model(seed=42)
        params = model.get_params()
        pruned, _ = prune_params(params, sparsity=0.5)

        for (W_orig, b_orig), (W_pruned, b_pruned) in zip(params, pruned):
            assert W_orig.shape == W_pruned.shape
            assert b_orig.shape == b_pruned.shape


class TestLatency:
    """Tests for latency measurement."""

    def test_measure_latency_returns_positive(self):
        model = create_full_model(seed=42)
        X = np.random.randn(1, 64)
        result = measure_inference_latency(model, X, num_runs=10)
        assert result['mean_latency_ms'] > 0
        assert result['std_latency_ms'] >= 0

    def test_edge_model_has_fewer_params(self):
        """Edge model should be smaller."""
        full = create_full_model(seed=42)
        edge = create_edge_model(seed=42)
        assert edge.count_params() < full.count_params()


class TestCompareModels:
    """Tests for model comparison."""

    def test_compare_returns_all_keys(self):
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        full = create_full_model(seed=42)
        edge = create_edge_model(seed=42)

        result = compare_models(full, edge, X_test, y_test)

        assert 'full' in result
        assert 'edge' in result
        assert 'full_quantized_int8' in result
        assert 'compression_ratio' in result
        assert result['compression_ratio'] > 1  # full is bigger
