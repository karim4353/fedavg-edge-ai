"""Small NumPy MLP for federated, edge-friendly classification.

Pure-NumPy implementation chosen on purpose:
- fully deterministic on CPU with a fixed seed,
- small enough for GitHub Actions runners,
- avoids a heavyweight DL framework (PyTorch / TF) — closer in spirit to
  the resource-constrained TinyML target devices in the team's presentation
  (Genann/CMSIS-NN-style minimal stacks).
"""
from __future__ import annotations

from typing import List

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


class MLP:
    """Two-layer MLP: Linear -> ReLU -> Linear -> Softmax."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 42) -> None:
        if min(n_in, n_hidden, n_out) <= 0:
            raise ValueError("layer sizes must be positive")
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        rng = np.random.default_rng(seed)
        # He initialization for ReLU
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)

    # ------------------------------------------------------------------ params
    def get_params(self) -> List[np.ndarray]:
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_params(self, params: List[np.ndarray]) -> None:
        if len(params) != 4:
            raise ValueError("expected 4 parameter tensors")
        self.W1 = params[0].astype(np.float32, copy=True)
        self.b1 = params[1].astype(np.float32, copy=True)
        self.W2 = params[2].astype(np.float32, copy=True)
        self.b2 = params[3].astype(np.float32, copy=True)

    def clone(self) -> "MLP":
        new = MLP(self.n_in, self.n_hidden, self.n_out)
        new.set_params(self.get_params())
        return new

    # ------------------------------------------------------------- forward/loss
    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)
        Z2 = A1 @ self.W2 + self.b2
        P = _softmax(Z2)
        return P, (X, Z1, A1, P)

    @staticmethod
    def cross_entropy(P: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.log(P[np.arange(len(y)), y] + eps).mean())

    # ------------------------------------------------------------------ update
    def backward(self, cache, y: np.ndarray) -> List[np.ndarray]:
        X, Z1, A1, P = cache
        n = len(y)
        dZ2 = P.copy()
        dZ2[np.arange(n), y] -= 1.0
        dZ2 /= n
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        return [dW1, db1, dW2, db2]

    def step(self, grads: List[np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads[0]
        self.b1 -= lr * grads[1]
        self.W2 -= lr * grads[2]
        self.b2 -= lr * grads[3]

    # ---------------------------------------------------------------- inference
    def predict(self, X: np.ndarray) -> np.ndarray:
        P, _ = self.forward(X)
        return P.argmax(axis=1)

    # ------------------------------------------------------------------- meta
    def num_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def size_bytes(self) -> int:
        """Storage assuming dense float32 weights."""
        return self.num_params() * 4
