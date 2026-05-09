"""
edge_optimizations.py - Edge AI optimizations for federated learning.

Implements realistic optimization techniques for deploying federated
learning on resource-constrained IoT devices (ESP32, Arduino, etc.),
matching the team's presentation focus on TinyML.

Optimizations implemented:
  1. Post-Training Quantization (float64 → float32 → int8)
     - Reduces model size by up to 8× (critical for devices with kB-level RAM)
     - Simulates what TensorFlow Lite does for embedded deployment
  2. Weight Pruning (magnitude-based)
     - Zeroes out small weights, reducing effective model complexity
     - Can be combined with sparse storage for memory savings
  3. Compact Model Architecture
     - Smaller hidden layers (48, 24 vs 128, 64)
     - ~60% parameter reduction while maintaining reasonable accuracy

These are the most practical Edge AI optimizations because:
  - The presentation highlights devices like ESP8266/ESP32 with very limited RAM
  - The paper focuses on communication efficiency; smaller models = fewer bytes
  - Quantization is the industry standard for embedded ML (TF Lite, CMSIS-NN)
"""

import numpy as np
import time
from src.models import MLPModel


def quantize_params(params, target_dtype='float32'):
    """Quantize model parameters to a lower precision.

    Simulates post-training quantization:
      float64 → float32: 2× size reduction (standard GPU precision)
      float64 → float16: 4× size reduction (mixed precision)
      float64 → int8: 8× size reduction (edge deployment standard)

    For int8, we use min-max quantization:
      q = round((x - x_min) / (x_max - x_min) * 255) - 128

    Args:
        params: list of (W, b) tuples (float64)
        target_dtype: 'float32', 'float16', or 'int8'

    Returns:
        quantized_params: list of (W, b) tuples in target dtype
        quant_info: dict with quantization metadata (for int8 dequantization)
    """
    quantized = []
    quant_info = {'scales': [], 'zero_points': [], 'dtype': target_dtype}

    for W, b in params:
        if target_dtype == 'int8':
            # Min-max symmetric quantization
            W_min, W_max = W.min(), W.max()
            b_min, b_max = b.min(), b.max()

            W_scale = (W_max - W_min) / 255.0 if W_max != W_min else 1.0
            b_scale = (b_max - b_min) / 255.0 if b_max != b_min else 1.0

            W_q = np.clip(np.round((W - W_min) / W_scale) - 128, -128, 127).astype(np.int8)
            b_q = np.clip(np.round((b - b_min) / b_scale) - 128, -128, 127).astype(np.int8)

            quantized.append((W_q, b_q))
            quant_info['scales'].append((W_scale, b_scale))
            quant_info['zero_points'].append((W_min, b_min))
        else:
            dtype = np.float32 if target_dtype == 'float32' else np.float16
            quantized.append((W.astype(dtype), b.astype(dtype)))
            quant_info['scales'].append((1.0, 1.0))
            quant_info['zero_points'].append((0.0, 0.0))

    return quantized, quant_info


def dequantize_params(quantized_params, quant_info):
    """Dequantize int8 parameters back to float64 for inference.

    Args:
        quantized_params: list of (W_q, b_q) tuples
        quant_info: dict from quantize_params

    Returns:
        params: list of (W, b) tuples in float64
    """
    params = []
    for i, (W_q, b_q) in enumerate(quantized_params):
        if quant_info['dtype'] == 'int8':
            W_scale, b_scale = quant_info['scales'][i]
            W_min, b_min = quant_info['zero_points'][i]
            W = (W_q.astype(np.float64) + 128) * W_scale + W_min
            b = (b_q.astype(np.float64) + 128) * b_scale + b_min
        else:
            W = W_q.astype(np.float64)
            b = b_q.astype(np.float64)
        params.append((W, b))
    return params


def prune_params(params, sparsity=0.3):
    """Magnitude-based weight pruning.

    Zeroes out the smallest weights (by absolute value) in each layer.
    This reduces the effective model complexity and can be combined
    with sparse storage formats for memory savings.

    Args:
        params: list of (W, b) tuples
        sparsity: fraction of weights to prune (0.0 = no pruning, 0.5 = 50%)

    Returns:
        pruned_params: list of (W, b) tuples with pruned weights
        pruning_stats: dict with per-layer pruning statistics
    """
    pruned = []
    stats = {'layers': [], 'total_pruned': 0, 'total_params': 0}

    for i, (W, b) in enumerate(params):
        # Compute threshold
        flat_w = np.abs(W.flatten())
        threshold = np.percentile(flat_w, sparsity * 100)

        # Create mask and apply
        mask = np.abs(W) >= threshold
        W_pruned = W * mask

        n_total = W.size
        n_pruned = n_total - np.count_nonzero(W_pruned)

        pruned.append((W_pruned, b.copy()))
        stats['layers'].append({
            'layer': i,
            'total': n_total,
            'pruned': n_pruned,
            'sparsity': n_pruned / n_total if n_total > 0 else 0,
        })
        stats['total_pruned'] += n_pruned
        stats['total_params'] += n_total

    stats['overall_sparsity'] = (
        stats['total_pruned'] / stats['total_params']
        if stats['total_params'] > 0 else 0
    )

    return pruned, stats


def measure_inference_latency(model, X, num_runs=100):
    """Measure average inference latency as a CPU time proxy.

    This simulates the latency that would be experienced on an edge device.
    While absolute times differ from embedded hardware, relative comparisons
    between model variants are meaningful.

    Args:
        model: MLPModel to benchmark
        X: input data (single sample or batch)
        num_runs: number of inference runs for averaging

    Returns:
        dict with 'mean_latency_ms', 'std_latency_ms', 'total_time_ms'
    """
    # Warm up
    for _ in range(5):
        model.forward(X)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.forward(X)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'total_time_ms': sum(latencies),
        'num_runs': num_runs,
    }


def estimate_memory_usage(model, dtype='float64'):
    """Estimate memory footprint of the model.

    Args:
        model: MLPModel
        dtype: assumed storage type

    Returns:
        dict with 'params_bytes', 'activations_estimate_bytes', 'total_bytes'
    """
    param_bytes = model.get_model_size_bytes(dtype)

    # Estimate activation memory for a single sample forward pass
    # (input + each hidden layer + output)
    dims = [model.input_dim] + list(model.hidden_dims) + [model.output_dim]
    nbytes = {'float64': 8, 'float32': 4, 'float16': 2, 'int8': 1}
    bytes_per_val = nbytes.get(dtype, 8)
    activation_bytes = sum(dims) * bytes_per_val

    return {
        'params_bytes': param_bytes,
        'activations_estimate_bytes': activation_bytes,
        'total_bytes': param_bytes + activation_bytes,
        'dtype': dtype,
    }


def compare_models(full_model, edge_model, X_test, y_test):
    """Compare full and edge models across multiple metrics.

    Args:
        full_model: the standard MLPModel
        edge_model: the compressed Edge MLPModel
        X_test, y_test: test data

    Returns:
        dict with comparison metrics
    """
    full_metrics = full_model.evaluate(X_test, y_test)
    edge_metrics = edge_model.evaluate(X_test, y_test)

    full_latency = measure_inference_latency(full_model, X_test[:1])
    edge_latency = measure_inference_latency(edge_model, X_test[:1])

    full_mem = estimate_memory_usage(full_model, 'float64')
    edge_mem = estimate_memory_usage(edge_model, 'float32')

    # Quantized versions
    full_q_params, full_q_info = quantize_params(full_model.get_params(), 'int8')
    full_q_mem = estimate_memory_usage(full_model, 'int8')

    return {
        'full': {
            'accuracy': full_metrics['accuracy'],
            'loss': full_metrics['loss'],
            'params': full_model.count_params(),
            'size_bytes': full_mem['params_bytes'],
            'latency_ms': full_latency['mean_latency_ms'],
        },
        'edge': {
            'accuracy': edge_metrics['accuracy'],
            'loss': edge_metrics['loss'],
            'params': edge_model.count_params(),
            'size_bytes': edge_mem['params_bytes'],
            'latency_ms': edge_latency['mean_latency_ms'],
        },
        'full_quantized_int8': {
            'params': full_model.count_params(),
            'size_bytes': full_q_mem['params_bytes'],
        },
        'compression_ratio': full_mem['params_bytes'] / edge_mem['params_bytes'] if edge_mem['params_bytes'] > 0 else 0,
    }
