"""
federated.py - Federated learning framework implementing FedSGD and FedAvg.

This module implements the core algorithms from McMahan et al. (2017),
"Communication-Efficient Learning of Deep Networks from Decentralized Data".

Algorithms implemented:
  - FedSGD: Federated SGD with B=∞ (full local dataset as one batch), E=1
  - FedAvg: Federated Averaging with configurable B (batch size) and E (local epochs)

Key parameters (from Algorithm 1 in the paper):
  - K: total number of clients
  - C: fraction of clients selected per round (0 < C <= 1)
  - E: number of local training epochs per round
  - B: local minibatch size
  - η: learning rate

The server aggregates using weighted averaging:
  w_{t+1} = Σ (n_k / m_t) * w_k_{t+1}
where m_t = Σ n_k for selected clients (erratum-corrected formula).
"""

import numpy as np
import time
from src.models import MLPModel, one_hot


def federated_sgd_round(global_model, client_data, learning_rate, client_fraction=1.0, seed=None):
    """Execute one round of FedSGD.

    FedSGD is FedAvg with B=∞ (full batch) and E=1.
    Each selected client computes one gradient step on its full local dataset,
    then the server averages the resulting models.

    Args:
        global_model: the current global MLPModel
        client_data: list of (X_k, y_k) tuples
        learning_rate: SGD learning rate η
        client_fraction: fraction C of clients to select
        seed: random seed for client selection

    Returns:
        updated global model parameters
    """
    return federated_avg_round(
        global_model, client_data, learning_rate,
        local_epochs=1, batch_size=None,  # B=∞, E=1
        client_fraction=client_fraction, seed=seed
    )


def federated_avg_round(global_model, client_data, learning_rate,
                         local_epochs=1, batch_size=10,
                         client_fraction=0.1, seed=None):
    """Execute one round of FedAvg (Algorithm 1 from the paper).

    Args:
        global_model: the current global MLPModel
        client_data: list of (X_k, y_k) tuples for all K clients
        learning_rate: SGD learning rate η
        local_epochs: E - number of local training epochs
        batch_size: B - local minibatch size (None = full batch)
        client_fraction: C - fraction of clients to select per round
        seed: random seed for client selection and shuffling

    Returns:
        new_params: aggregated model parameters
    """
    rng = np.random.RandomState(seed)
    K = len(client_data)

    # Select m = max(C*K, 1) clients
    m = max(int(client_fraction * K), 1)
    selected = rng.choice(K, size=m, replace=False)

    # Collect updated parameters from each selected client
    client_params = []
    client_sizes = []

    global_params = global_model.get_params()

    for k in selected:
        X_k, y_k = client_data[k]
        n_k = len(y_k)

        # Create a local copy of the model with global parameters
        local_model = MLPModel(
            input_dim=global_model.input_dim,
            hidden_dims=global_model.hidden_dims,
            output_dim=global_model.output_dim,
            seed=0,  # seed doesn't matter, we overwrite params
        )
        local_model.set_params(global_params)

        # Client local training (ClientUpdate in Algorithm 1)
        _client_update(local_model, X_k, y_k, learning_rate,
                       local_epochs, batch_size, rng)

        client_params.append(local_model.get_params())
        client_sizes.append(n_k)

    # Server aggregation: weighted average by number of samples
    new_params = _aggregate_params(client_params, client_sizes)
    return new_params


def _client_update(model, X, y, learning_rate, local_epochs, batch_size, rng):
    """Perform local SGD training on a client (ClientUpdate in Algorithm 1).

    Args:
        model: local MLPModel to train
        X: client's training features
        y: client's training labels
        learning_rate: η
        local_epochs: E
        batch_size: B (None for full batch)
        rng: numpy RandomState for shuffling
    """
    n = len(y)
    y_onehot = one_hot(y, model.output_dim)

    for epoch in range(local_epochs):
        if batch_size is None or batch_size >= n:
            # Full batch (FedSGD mode: B=∞)
            model.forward(X)
            model.backward(y_onehot, learning_rate)
        else:
            # Mini-batch SGD
            indices = rng.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]
                model.forward(X_batch)
                model.backward(y_batch, learning_rate)


def _aggregate_params(client_params, client_sizes):
    """Weighted average of model parameters (server aggregation).

    Implements: w_{t+1} = Σ (n_k / m_t) * w_k_{t+1}
    where m_t = Σ n_k for all selected clients.

    Args:
        client_params: list of parameter lists from each client
        client_sizes: list of sample counts n_k for each client

    Returns:
        aggregated parameters as list of (W, b) tuples
    """
    total_samples = sum(client_sizes)

    # Initialize with zeros
    num_layers = len(client_params[0])
    new_params = []
    for layer_idx in range(num_layers):
        W_shape = client_params[0][layer_idx][0].shape
        b_shape = client_params[0][layer_idx][1].shape
        new_params.append((np.zeros(W_shape), np.zeros(b_shape)))

    # Weighted sum
    for params, n_k in zip(client_params, client_sizes):
        weight = n_k / total_samples
        for layer_idx in range(num_layers):
            new_params[layer_idx] = (
                new_params[layer_idx][0] + weight * params[layer_idx][0],
                new_params[layer_idx][1] + weight * params[layer_idx][1],
            )

    return new_params


def train_federated(model, client_data, X_test, y_test,
                    num_rounds=50, learning_rate=0.1,
                    local_epochs=1, batch_size=10,
                    client_fraction=0.1, seed=42,
                    algorithm='fedavg', verbose=True):
    """Full federated training loop.

    Args:
        model: MLPModel to train
        client_data: list of (X_k, y_k) tuples
        X_test, y_test: test data for evaluation
        num_rounds: number of communication rounds T
        learning_rate: η
        local_epochs: E
        batch_size: B (None for FedSGD full batch)
        client_fraction: C
        seed: random seed
        algorithm: 'fedavg' or 'fedsgd'
        verbose: whether to print progress

    Returns:
        history: dict with 'round', 'accuracy', 'loss', 'time' lists
    """
    history = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'time': [],
    }

    start_time = time.time()

    for t in range(1, num_rounds + 1):
        round_seed = seed + t  # deterministic per-round seed

        if algorithm == 'fedsgd':
            new_params = federated_sgd_round(
                model, client_data, learning_rate,
                client_fraction=client_fraction, seed=round_seed
            )
        else:  # fedavg
            new_params = federated_avg_round(
                model, client_data, learning_rate,
                local_epochs=local_epochs, batch_size=batch_size,
                client_fraction=client_fraction, seed=round_seed
            )

        model.set_params(new_params)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        elapsed = time.time() - start_time

        history['round'].append(t)
        history['accuracy'].append(metrics['accuracy'])
        history['loss'].append(metrics['loss'])
        history['time'].append(elapsed)

        if verbose and (t % max(1, num_rounds // 10) == 0 or t == 1):
            print(f"  Round {t:4d}/{num_rounds}: "
                  f"acc={metrics['accuracy']:.4f}, "
                  f"loss={metrics['loss']:.4f}, "
                  f"time={elapsed:.2f}s")

    return history


def train_centralized(model, X_train, y_train, X_test, y_test,
                      num_epochs=50, learning_rate=0.1, batch_size=32,
                      seed=42, verbose=True):
    """Centralized (non-federated) training baseline.

    Standard mini-batch SGD on the full dataset.

    Args:
        model: MLPModel to train
        X_train, y_train: training data
        X_test, y_test: test data
        num_epochs: number of training epochs
        learning_rate: SGD learning rate
        batch_size: minibatch size
        seed: random seed
        verbose: print progress

    Returns:
        history: dict with 'epoch', 'accuracy', 'loss', 'time' lists
    """
    rng = np.random.RandomState(seed)
    n = len(y_train)
    y_train_onehot = one_hot(y_train, model.output_dim)

    history = {
        'epoch': [],
        'accuracy': [],
        'loss': [],
        'time': [],
    }

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Shuffle
        indices = rng.permutation(n)

        # Mini-batch SGD
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            X_batch = X_train[batch_idx]
            y_batch = y_train_onehot[batch_idx]
            model.forward(X_batch)
            model.backward(y_batch, learning_rate)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        elapsed = time.time() - start_time

        history['epoch'].append(epoch)
        history['accuracy'].append(metrics['accuracy'])
        history['loss'].append(metrics['loss'])
        history['time'].append(elapsed)

        if verbose and (epoch % max(1, num_epochs // 10) == 0 or epoch == 1):
            print(f"  Epoch {epoch:4d}/{num_epochs}: "
                  f"acc={metrics['accuracy']:.4f}, "
                  f"loss={metrics['loss']:.4f}, "
                  f"time={elapsed:.2f}s")

    return history
