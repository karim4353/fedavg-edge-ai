"""Classification metrics implemented in NumPy (no sklearn dep)."""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over the explicit class set 0..n_classes-1.

    Classes that never appear in either `y_true` or `y_pred` contribute 0 to
    the average, which is the conservative behavior expected on heavily
    imbalanced data such as the ECG heartbeat dataset.
    """
    f1s: list[float] = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0.0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def cross_entropy(P: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.log(P[np.arange(len(y)), y] + eps).mean())
