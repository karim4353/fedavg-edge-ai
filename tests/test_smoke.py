"""
test_smoke.py - End-to-end smoke test that runs a tiny experiment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.data import load_digit_data, partition_iid, partition_non_iid
from src.models import create_full_model, create_edge_model
from src.federated import train_federated, train_centralized
from src.edge_optimizations import quantize_params, dequantize_params, prune_params
from src.mqtt_simulation import MQTTSimulator


class TestEndToEnd:
    """End-to-end smoke tests."""

    def test_full_pipeline_runs(self):
        """A minimal experiment should run without errors."""
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        clients = partition_iid(X_train, y_train, 3, seed=42)

        # Centralized
        model = create_full_model(seed=42)
        hist = train_centralized(
            model, X_train, y_train, X_test, y_test,
            num_epochs=2, learning_rate=0.1, seed=42, verbose=False
        )
        assert hist['accuracy'][-1] > 0

        # FedAvg
        model2 = create_full_model(seed=42)
        hist2 = train_federated(
            model2, clients, X_test, y_test,
            num_rounds=3, learning_rate=0.1,
            local_epochs=2, batch_size=10,
            seed=42, verbose=False
        )
        assert hist2['accuracy'][-1] > 0

    def test_edge_pipeline_runs(self):
        """Edge model training should work end-to-end."""
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        clients = partition_iid(X_train, y_train, 3, seed=42)

        model = create_edge_model(seed=42)
        hist = train_federated(
            model, clients, X_test, y_test,
            num_rounds=3, learning_rate=0.1,
            local_epochs=2, batch_size=10,
            seed=42, verbose=False
        )
        assert len(hist['round']) == 3
        assert hist['accuracy'][-1] > 0

    def test_non_iid_pipeline_runs(self):
        """Non-IID partition should work with FedAvg."""
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        clients = partition_non_iid(X_train, y_train, 5, seed=42)

        model = create_full_model(seed=42)
        hist = train_federated(
            model, clients, X_test, y_test,
            num_rounds=3, learning_rate=0.1,
            local_epochs=2, batch_size=10,
            seed=42, verbose=False
        )
        assert len(hist['round']) == 3

    def test_quantize_after_training(self):
        """Post-training quantization should work on a trained model."""
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        model = create_full_model(seed=42)
        train_centralized(
            model, X_train, y_train, X_test, y_test,
            num_epochs=2, learning_rate=0.1, seed=42, verbose=False
        )

        # Quantize
        params = model.get_params()
        q_params, q_info = quantize_params(params, 'int8')
        deq_params = dequantize_params(q_params, q_info)
        model.set_params(deq_params)

        acc = model.evaluate(X_test, y_test)['accuracy']
        assert acc > 0  # should still produce some predictions

    def test_prune_after_training(self):
        """Post-training pruning should work."""
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        model = create_full_model(seed=42)
        train_centralized(
            model, X_train, y_train, X_test, y_test,
            num_epochs=2, learning_rate=0.1, seed=42, verbose=False
        )

        params = model.get_params()
        pruned, stats = prune_params(params, sparsity=0.3)
        model.set_params(pruned)

        acc = model.evaluate(X_test, y_test)['accuracy']
        assert acc > 0

    def test_mqtt_simulation(self):
        """MQTT simulation should track communication."""
        mqtt = MQTTSimulator(num_clients=3)
        model = create_full_model(seed=42)
        params = model.get_params()

        mqtt.publish_global_model(params, round_num=0)
        for c in range(3):
            mqtt.publish_client_update(params, c, round_num=0)

        stats = mqtt.get_communication_stats()
        assert stats['total_bytes'] > 0
        assert stats['total_messages'] == 6  # 3 broadcasts + 3 updates

    def test_deterministic_full_run(self):
        """Two identical runs should produce the same final accuracy."""
        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        clients = partition_iid(X_train, y_train, 3, seed=42)

        model1 = create_full_model(seed=42)
        hist1 = train_federated(
            model1, clients, X_test, y_test,
            num_rounds=5, learning_rate=0.1,
            local_epochs=2, batch_size=10,
            seed=42, verbose=False
        )

        model2 = create_full_model(seed=42)
        hist2 = train_federated(
            model2, clients, X_test, y_test,
            num_rounds=5, learning_rate=0.1,
            local_epochs=2, batch_size=10,
            seed=42, verbose=False
        )

        np.testing.assert_array_equal(hist1['accuracy'], hist2['accuracy'])
        np.testing.assert_array_equal(hist1['loss'], hist2['loss'])


class TestOutputGeneration:
    """Tests that verify output file generation."""

    def test_smoke_experiment_generates_files(self, tmp_path):
        """Smoke experiment should generate CSV and PNG files."""
        import pandas as pd

        X_train, X_test, y_train, y_test = load_digit_data(seed=42)
        clients = partition_iid(X_train, y_train, 3, seed=42)

        model = create_full_model(seed=42)
        hist = train_federated(
            model, clients, X_test, y_test,
            num_rounds=3, learning_rate=0.1,
            local_epochs=1, batch_size=10,
            seed=42, verbose=False
        )

        # Save CSV
        csv_path = tmp_path / 'test_results.csv'
        df = pd.DataFrame({
            'round': hist['round'],
            'accuracy': hist['accuracy'],
            'loss': hist['loss'],
        })
        df.to_csv(csv_path, index=False)
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

        # Verify CSV content
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 3
        assert 'accuracy' in loaded.columns

    def test_plot_generation(self, tmp_path):
        """Plot generation should produce valid PNG files."""
        from src.visualization import plot_accuracy_curves

        history = {
            'round': [1, 2, 3],
            'accuracy': [0.3, 0.5, 0.7],
            'loss': [2.0, 1.5, 1.0],
        }

        plot_path = str(tmp_path / 'test_plot.png')
        plot_accuracy_curves({'Test': history}, plot_path, title='Test')

        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 0
