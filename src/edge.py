"""Edge AI optimizations.

Three resource-aware techniques that target the constraints listed in the
team's presentation (RAM in the kB range, weak CPUs, no float64):

1. Post-training INT8 weight quantization (per-tensor symmetric).
2. Magnitude pruning of weight matrices (sparsity sweep).
3. Compact-width variant of the MLP, trainable end-to-end.

All three operate on the NumPy MLP from `src.model` and produce *new* model
instances; the original model is left untouched. Size estimates are reported
under realistic on-device storage formats (raw int8 for quantized models,
CSR-style indices+values for pruned ones).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .model import MLP

WEIGHT_INDICES = (0, 2)  # indices of W1 and W2 in get_params()


# --------------------------------------------------------------- quantization
def quantize_int8(model: MLP) -> MLP:
    """Symmetric per-tensor INT8 quantization, returned as fp32 (simulated dequant).

    For each parameter tensor `p`, we compute scale `s = max(|p|) / 127`,
    round to int8, then re-cast to fp32. The returned model behaves exactly
    like a real int8 deployment would after dequantization on-device.
    """
    new_model = MLP(model.n_in, model.n_hidden, model.n_out)
    new_params: List[np.ndarray] = []
    for p in model.get_params():
        max_abs = float(np.max(np.abs(p)))
        if max_abs <= 0.0:
            new_params.append(p.astype(np.float32, copy=True))
            continue
        scale = max_abs / 127.0
        q = np.round(p / scale).clip(-127, 127).astype(np.int8)
        deq = (q.astype(np.float32) * np.float32(scale)).astype(np.float32)
        new_params.append(deq)
    new_model.set_params(new_params)
    return new_model


def quantized_size_bytes(model: MLP) -> int:
    """On-device storage assuming int8 weights + fp32 per-tensor scales."""
    bytes_total = 0
    for p in model.get_params():
        bytes_total += p.size * 1  # int8 values
        bytes_total += 4            # fp32 scale per tensor
    return bytes_total


# --------------------------------------------------------------------- pruning
def magnitude_prune(model: MLP, sparsity: float) -> MLP:
    """Zero out the smallest-magnitude `sparsity` fraction of each weight matrix.

    Biases are never pruned. `sparsity == 0` returns an exact copy.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1)")
    new_model = MLP(model.n_in, model.n_hidden, model.n_out)
    params = model.get_params()
    new_params: List[np.ndarray] = []
    for i, p in enumerate(params):
        if i in WEIGHT_INDICES and sparsity > 0.0 and p.size > 0:
            flat = np.abs(p).ravel()
            k = int(sparsity * flat.size)
            if k > 0:
                threshold = np.partition(flat, k - 1)[k - 1]
                mask = (np.abs(p) > threshold).astype(np.float32)
                new_params.append((p * mask).astype(np.float32))
            else:
                new_params.append(p.astype(np.float32, copy=True))
        else:
            new_params.append(p.astype(np.float32, copy=True))
    new_model.set_params(new_params)
    return new_model


def model_sparsity(model: MLP) -> float:
    """Fraction of zero weights across W1 and W2 (excludes biases)."""
    total = 0
    zero = 0
    for p in (model.W1, model.W2):
        total += p.size
        zero += int(np.sum(p == 0))
    return zero / total if total > 0 else 0.0


def pruned_size_bytes(model: MLP) -> int:
    """CSR-style storage estimate for a sparse on-device deployment.

    For every weight matrix: nonzero values (fp32) + column indices (int16) +
    row pointers (int32). Biases stay dense fp32.
    """
    bytes_total = 0
    # weight matrices
    for p in (model.W1, model.W2):
        nnz = int(np.count_nonzero(p))
        bytes_total += nnz * 4         # fp32 values
        bytes_total += nnz * 2         # int16 column indices
        bytes_total += (p.shape[0] + 1) * 4  # int32 row pointers
    # biases (always dense)
    bytes_total += model.b1.size * 4
    bytes_total += model.b2.size * 4
    return bytes_total


# --------------------------------------------------- compact-width MLP variant
def compact_mlp(model: MLP, factor: int = 2, seed: int = 42) -> MLP:
    """Build a smaller-width sibling by dividing the hidden dimension by `factor`.

    Used to *train from scratch* a more memory-friendly model — orthogonal to
    quantization/pruning, which compress an already-trained network.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    new_hidden = max(1, model.n_hidden // factor)
    return MLP(model.n_in, new_hidden, model.n_out, seed=seed)
