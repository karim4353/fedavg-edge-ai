"""Edge AI optimization tests: quantization, pruning, compact variant."""
import numpy as np

from src.edge import (
    compact_mlp,
    magnitude_prune,
    model_sparsity,
    pruned_size_bytes,
    quantize_int8,
    quantized_size_bytes,
)
from src.model import MLP


def test_quantization_keeps_shapes_and_dtype():
    m = MLP(8, 8, 4, seed=0)
    q = quantize_int8(m)
    for p_orig, p_q in zip(m.get_params(), q.get_params()):
        assert p_q.shape == p_orig.shape
        assert p_q.dtype == np.float32


def test_quantization_reduces_size():
    m = MLP(32, 32, 5, seed=0)
    assert quantized_size_bytes(m) < m.size_bytes()


def test_quantization_close_to_original():
    """Dequantized weights should approximate the originals."""
    m = MLP(16, 16, 4, seed=0)
    q = quantize_int8(m)
    for p_orig, p_q in zip(m.get_params(), q.get_params()):
        max_abs = float(np.max(np.abs(p_orig))) + 1e-12
        scale = max_abs / 127.0
        # Quantization error per element <= scale (rounding to nearest int8 step)
        assert np.max(np.abs(p_orig - p_q)) <= scale + 1e-6


def test_pruning_zero_sparsity_is_identity():
    m = MLP(8, 8, 4, seed=0)
    p = magnitude_prune(m, sparsity=0.0)
    for a, b in zip(m.get_params(), p.get_params()):
        np.testing.assert_array_equal(a, b)


def test_pruning_increases_sparsity():
    m = MLP(16, 16, 5, seed=0)
    pruned = magnitude_prune(m, sparsity=0.5)
    sp = model_sparsity(pruned)
    assert 0.4 <= sp <= 0.6


def test_pruning_preserves_biases():
    m = MLP(8, 8, 4, seed=0)
    pruned = magnitude_prune(m, sparsity=0.7)
    np.testing.assert_array_equal(m.b1, pruned.b1)
    np.testing.assert_array_equal(m.b2, pruned.b2)


def test_pruned_size_smaller_at_high_sparsity():
    m = MLP(64, 64, 5, seed=0)
    pruned = magnitude_prune(m, sparsity=0.9)
    assert pruned_size_bytes(pruned) < m.size_bytes()


def test_compact_mlp_smaller():
    m = MLP(32, 32, 5, seed=0)
    s = compact_mlp(m, factor=2)
    assert s.n_hidden == 16
    assert s.num_params() < m.num_params()
