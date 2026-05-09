"""
models.py - Lightweight neural network models for federated learning.

Implements simple MLP classifiers using only NumPy (no PyTorch/TensorFlow),
making them suitable for Edge AI / embedded deployment simulation.

Model architectures:
  - MLP_Full: 2-hidden-layer MLP (64 -> 128 -> 64 -> 10)
    Inspired by the paper's "2NN" (200-unit hidden layers) but scaled down
    for the smaller digits dataset (64 features vs 784 for MNIST).
  - MLP_Edge: Compressed model (64 -> 48 -> 24 -> 10)
    Simulates a resource-constrained Edge AI deployment with ~60% fewer parameters.

Both models use ReLU activations and softmax output, matching the paper's
architecture choices (Section 3).
"""

import numpy as np
import sys


def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU."""
    return (z > 0).astype(float)


def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true_onehot):
    """Cross-entropy loss."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))


def one_hot(y, num_classes=10):
    """Convert labels to one-hot encoding."""
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh


class MLPModel:
    """Multi-Layer Perceptron with configurable hidden layer sizes.

    This is a pure NumPy implementation suitable for Edge AI simulation.
    Supports forward pass, backpropagation, and SGD training.
    """

    def __init__(self, input_dim=64, hidden_dims=(128, 64), output_dim=10, seed=42):
        """Initialize MLP with Xavier/He initialization.

        Args:
            input_dim: number of input features
            hidden_dims: tuple of hidden layer sizes
            output_dim: number of output classes
            seed: random seed for weight initialization
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        rng = np.random.RandomState(seed)

        # Build layers
        self.weights = []
        self.biases = []
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        for i in range(len(dims) - 1):
            # He initialization for ReLU
            scale = np.sqrt(2.0 / dims[i])
            W = rng.randn(dims[i], dims[i + 1]) * scale
            b = np.zeros((1, dims[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass through the network.

        Args:
            X: input data (batch_size, input_dim)

        Returns:
            output probabilities (batch_size, output_dim)
        """
        self.activations = [X]
        self.z_values = []

        current = X
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            current = relu(z)
            self.activations.append(current)

        # Output layer with softmax
        z_out = current @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z_out)
        output = softmax(z_out)
        self.activations.append(output)

        return output

    def backward(self, y_true_onehot, learning_rate):
        """Backpropagation with SGD update.

        Args:
            y_true_onehot: one-hot encoded true labels
            learning_rate: SGD learning rate η

        Returns:
            gradients as list of (dW, db) tuples
        """
        m = y_true_onehot.shape[0]
        gradients = []

        # Output layer gradient (softmax + cross-entropy)
        delta = self.activations[-1] - y_true_onehot

        for i in range(len(self.weights) - 1, -1, -1):
            dW = (self.activations[i].T @ delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            gradients.insert(0, (dW, db))

            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(self.z_values[i - 1])

        # Apply updates
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

        return gradients

    def predict(self, X):
        """Predict class labels.

        Args:
            X: input data

        Returns:
            predicted class labels
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        """Evaluate accuracy and loss.

        Args:
            X: input features
            y: true labels (integer)

        Returns:
            dict with 'accuracy', 'loss', 'predictions'
        """
        probs = self.forward(X)
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y)
        loss = cross_entropy_loss(probs, one_hot(y, self.output_dim))
        return {
            'accuracy': accuracy,
            'loss': loss,
            'predictions': predictions,
        }

    def get_params(self):
        """Get a deep copy of all model parameters.

        Returns:
            list of (W_copy, b_copy) tuples
        """
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append((W.copy(), b.copy()))
        return params

    def set_params(self, params):
        """Set model parameters from a list of (W, b) tuples.

        Args:
            params: list of (W, b) tuples
        """
        for i, (W, b) in enumerate(params):
            self.weights[i] = W.copy()
            self.biases[i] = b.copy()

    def count_params(self):
        """Count total number of trainable parameters.

        Returns:
            int: total parameter count
        """
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total

    def get_model_size_bytes(self, dtype='float64'):
        """Estimate model size in bytes.

        Args:
            dtype: data type string ('float64', 'float32', 'float16', 'int8')

        Returns:
            int: estimated size in bytes
        """
        nbytes = {'float64': 8, 'float32': 4, 'float16': 2, 'int8': 1}
        return self.count_params() * nbytes.get(dtype, 8)


def create_full_model(seed=42):
    """Create the full-size MLP model.

    Architecture: 64 -> 128 -> 64 -> 10
    Inspired by the paper's 2NN but scaled for the digits dataset.
    """
    return MLPModel(input_dim=64, hidden_dims=(128, 64), output_dim=10, seed=seed)


def create_edge_model(seed=42):
    """Create a compressed Edge AI model.

    Architecture: 64 -> 48 -> 24 -> 10
    ~60% fewer parameters than the full model, suitable for
    resource-constrained devices (ESP32, Arduino, etc.).
    """
    return MLPModel(input_dim=64, hidden_dims=(48, 24), output_dim=10, seed=seed)
