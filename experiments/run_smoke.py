"""
run_smoke.py - Smoke test experiment for CI/CD validation.

Runs a minimal version of all experiments to verify:
  1. Code runs without errors
  2. Output files (CSV, PNG) are generated
  3. Results are deterministic

Expected runtime: ~10-30 seconds on CPU.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
import time

from src.data import load_digit_data, partition_iid, partition_non_iid, get_client_stats
from src.models import create_full_model, create_edge_model
from src.federated import train_federated, train_centralized
from src.edge_optimizations import (
    quantize_params, dequantize_params, prune_params,
    measure_inference_latency, compare_models
)
from src.visualization import plot_accuracy_curves, plot_loss_curves, plot_edge_comparison
from src.mqtt_simulation import MQTTSimulator


def run_smoke_experiment():
    """Run the smoke test experiment."""
    print("=" * 70)
    print("SMOKE TEST EXPERIMENT")
    print("=" * 70)

    start_time = time.time()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'smoke.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = config['experiment']['seed']
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'smoke')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print("\n[1/6] Loading data...")
    X_train, X_test, y_train, y_test = load_digit_data(seed=seed)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Centralized baseline
    print("\n[2/6] Centralized baseline...")
    model_cent = create_full_model(seed=seed)
    hist_cent = train_centralized(
        model_cent, X_train, y_train, X_test, y_test,
        num_epochs=config['centralized']['num_epochs'],
        learning_rate=config['centralized']['learning_rate'],
        batch_size=config['centralized']['batch_size'],
        seed=seed, verbose=True
    )

    # 3. FedSGD (IID)
    print("\n[3/6] FedSGD (IID)...")
    client_data_iid = partition_iid(X_train, y_train,
                                      config['data']['num_clients'], seed=seed)
    model_sgd = create_full_model(seed=seed)
    hist_sgd = train_federated(
        model_sgd, client_data_iid, X_test, y_test,
        num_rounds=config['fedsgd']['num_rounds'],
        learning_rate=config['fedsgd']['learning_rate'],
        client_fraction=config['fedsgd']['client_fraction'],
        local_epochs=1, batch_size=None,
        seed=seed, algorithm='fedsgd', verbose=True
    )

    # 4. FedAvg (IID)
    print("\n[4/6] FedAvg (IID)...")
    model_avg = create_full_model(seed=seed)
    hist_avg = train_federated(
        model_avg, client_data_iid, X_test, y_test,
        num_rounds=config['fedavg']['num_rounds'],
        learning_rate=config['fedavg']['learning_rate'],
        local_epochs=config['fedavg']['local_epochs'],
        batch_size=config['fedavg']['batch_size'],
        client_fraction=config['fedavg']['client_fraction'],
        seed=seed, algorithm='fedavg', verbose=True
    )

    # 5. Edge AI variant
    print("\n[5/6] Edge AI FedAvg (IID)...")
    model_edge = create_edge_model(seed=seed)
    hist_edge = train_federated(
        model_edge, client_data_iid, X_test, y_test,
        num_rounds=config['edge']['num_rounds'],
        learning_rate=config['edge']['learning_rate'],
        local_epochs=config['edge']['local_epochs'],
        batch_size=config['edge']['batch_size'],
        client_fraction=config['edge']['client_fraction'],
        seed=seed, algorithm='fedavg', verbose=True
    )

    # 6. MQTT simulation
    print("\n[6/6] MQTT communication simulation...")
    mqtt = MQTTSimulator(num_clients=config['data']['num_clients'])
    for r in range(config['fedavg']['num_rounds']):
        mqtt.publish_global_model(model_avg.get_params(), r)
        for c in range(config['data']['num_clients']):
            mqtt.publish_client_update(model_avg.get_params(), c, r)
    comm_stats = mqtt.get_communication_stats()
    print(f"  Total data transferred: {comm_stats['total_kb']:.1f} KB")
    print(f"  Total messages: {comm_stats['total_messages']}")

    # Generate outputs
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    # Plots
    results = {
        'Centralized': hist_cent,
        'FedSGD': hist_sgd,
        'FedAvg': hist_avg,
        'Edge FedAvg': hist_edge,
    }
    plot_accuracy_curves(results, os.path.join(output_dir, 'accuracy_curves.png'),
                         title='Smoke Test: Accuracy Comparison')
    plot_loss_curves(results, os.path.join(output_dir, 'loss_curves.png'),
                     title='Smoke Test: Loss Comparison')

    # Edge comparison
    comparison = compare_models(model_avg, model_edge, X_test, y_test)
    plot_edge_comparison(comparison, os.path.join(output_dir, 'edge_comparison.png'))

    # CSV results
    rows = []
    for name, hist in results.items():
        x_key = 'round' if 'round' in hist else 'epoch'
        final_acc = hist['accuracy'][-1]
        final_loss = hist['loss'][-1]
        total_time = hist['time'][-1]
        rows.append({
            'Algorithm': name,
            'Final Accuracy': f"{final_acc:.4f}",
            'Final Loss': f"{final_loss:.4f}",
            'Total Time (s)': f"{total_time:.2f}",
            'Rounds/Epochs': len(hist[x_key]),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    elapsed = time.time() - start_time

    log_path = os.path.join(output_dir, 'experiment_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Smoke Test Experiment Log\n")
        f.write("=========================\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Clients: {config['data']['num_clients']}\n")
        f.write(f"Elapsed seconds: {elapsed:.2f}\n")
        f.write(f"MQTT total KB: {comm_stats['total_kb']:.1f}\n")
        f.write(f"MQTT total messages: {comm_stats['total_messages']}\n\n")
        f.write("Results summary:\n")
        f.write(df.to_string(index=False))
        f.write("\n\nGenerated files:\n")
        for filename in [
            'accuracy_curves.png',
            'loss_curves.png',
            'edge_comparison.png',
            'results_summary.csv',
        ]:
            f.write(f"- {filename}\n")
    print(f"  Saved: {log_path}")

    print(f"\n[OK] Smoke test completed in {elapsed:.1f}s")
    print(f"  Output directory: {output_dir}")

    return df


if __name__ == '__main__':
    run_smoke_experiment()
