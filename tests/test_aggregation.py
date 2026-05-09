"""Aggregation correctness: server-side weighted averaging."""
import numpy as np

from src.server import weighted_average


def test_uniform_weights_match_simple_mean():
    a = [np.ones((2, 3), dtype=np.float32), np.zeros((4,), dtype=np.float32)]
    b = [np.full((2, 3), 3.0, dtype=np.float32), np.full((4,), 2.0, dtype=np.float32)]
    avg = weighted_average([a, b], weights=[1, 1])
    assert avg[0].shape == (2, 3)
    np.testing.assert_allclose(avg[0], np.full((2, 3), 2.0))
    np.testing.assert_allclose(avg[1], np.full((4,), 1.0))


def test_weighted_proportional_to_sample_count():
    a = [np.full((1,), 1.0, dtype=np.float32)]
    b = [np.full((1,), 7.0, dtype=np.float32)]
    avg = weighted_average([a, b], weights=[300, 100])
    expected = (300 * 1.0 + 100 * 7.0) / 400.0
    np.testing.assert_allclose(avg[0], expected, atol=1e-6)


def test_returns_float32():
    a = [np.ones((3,), dtype=np.float64)]
    b = [np.zeros((3,), dtype=np.float64)]
    avg = weighted_average([a, b], weights=[1, 3])
    assert avg[0].dtype == np.float32


def test_single_client_passthrough():
    a = [np.array([1.0, -2.0, 3.0], dtype=np.float32)]
    avg = weighted_average([a], weights=[42])
    np.testing.assert_allclose(avg[0], a[0])


def test_rejects_empty_input():
    import pytest
    with pytest.raises(ValueError):
        weighted_average([], weights=[])
