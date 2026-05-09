"""Determinism: with a fixed seed, repeated runs produce identical results."""
import numpy as np

from src.data import make_synthetic_classification, partition_iid, train_test_split
from src.federated import run_federated
from src.metrics import accuracy
from src.model import MLP
from src.utils import set_seed


def _run(seed: int = 0):
    set_seed(seed)
    X, y = make_synthetic_classification(n_samples=200, n_features=8, n_classes=4, seed=seed)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_frac=0.25, seed=seed)
    clients = partition_iid(X_tr, y_tr, n_clients=3, seed=seed)
    model = MLP(8, 8, 4, seed=seed)
    run_federated(model, clients, n_rounds=3, lr=0.05, local_epochs=2,
                  batch_size=8, seed=seed)
    return accuracy(y_te, model.predict(X_te)), model.get_params()


def test_same_seed_identical_accuracy():
    acc_a, _ = _run(seed=123)
    acc_b, _ = _run(seed=123)
    assert acc_a == acc_b


def test_same_seed_identical_weights():
    _, params_a = _run(seed=7)
    _, params_b = _run(seed=7)
    for pa, pb in zip(params_a, params_b):
        np.testing.assert_array_equal(pa, pb)


def test_different_seeds_diverge():
    _, params_a = _run(seed=1)
    _, params_b = _run(seed=2)
    # At least one tensor should differ - else seeding is broken
    diffs = [not np.array_equal(a, b) for a, b in zip(params_a, params_b)]
    assert any(diffs)
