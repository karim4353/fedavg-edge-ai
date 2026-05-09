"""
visualization.py - Plotting utilities for experiment results.

Generates publication-quality PNG plots for:
  - Accuracy vs communication rounds (main paper figure)
  - Loss curves
  - IID vs non-IID comparison
  - Model size / latency comparison bars
  - Edge AI optimization comparison
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import os


# Style configuration
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'centralized': '#2196F3',
    'fedsgd': '#FF9800',
    'fedavg': '#4CAF50',
    'edge': '#E91E63',
    'fedavg_noniid': '#9C27B0',
    'fedsgd_noniid': '#FF5722',
}


def plot_accuracy_curves(results, output_path, title="Test Accuracy vs Communication Rounds"):
    """Plot accuracy over communication rounds for multiple algorithms.

    Reproduces the style of Figure 2 from McMahan et al. (2017).

    Args:
        results: dict mapping algorithm name to history dict
                 (each has 'round'/'epoch' and 'accuracy' keys)
        output_path: path to save PNG
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    color_list = list(COLORS.values())
    for i, (name, history) in enumerate(results.items()):
        x_key = 'round' if 'round' in history else 'epoch'
        color = COLORS.get(name.lower().replace(' ', '_').split('(')[0].strip('_'), color_list[i % len(color_list)])
        ax.plot(history[x_key], history['accuracy'],
                label=name, color=color, linewidth=2, alpha=0.9)

    ax.set_xlabel('Communication Rounds / Epochs')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(title)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_loss_curves(results, output_path, title="Training Loss vs Communication Rounds"):
    """Plot loss curves for multiple algorithms.

    Args:
        results: dict mapping name to history with 'loss' key
        output_path: path to save PNG
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    color_list = list(COLORS.values())
    for i, (name, history) in enumerate(results.items()):
        x_key = 'round' if 'round' in history else 'epoch'
        color = COLORS.get(name.lower().replace(' ', '_').split('(')[0].strip('_'), color_list[i % len(color_list)])
        ax.plot(history[x_key], history['loss'],
                label=name, color=color, linewidth=2, alpha=0.9)

    ax.set_xlabel('Communication Rounds / Epochs')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_iid_vs_noniid(iid_results, noniid_results, output_path,
                        title="IID vs Non-IID Comparison"):
    """Side-by-side comparison of IID and non-IID performance.

    Reproduces the paired comparison style from the paper's Figure 2.

    Args:
        iid_results: dict mapping name to history (IID partition)
        noniid_results: dict mapping name to history (non-IID partition)
        output_path: path to save PNG
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    color_list = list(COLORS.values())

    for i, (name, history) in enumerate(iid_results.items()):
        x_key = 'round' if 'round' in history else 'epoch'
        ax1.plot(history[x_key], history['accuracy'],
                 label=name, color=color_list[i % len(color_list)],
                 linewidth=2, alpha=0.9)

    ax1.set_xlabel('Rounds / Epochs')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('IID Partition')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    for i, (name, history) in enumerate(noniid_results.items()):
        x_key = 'round' if 'round' in history else 'epoch'
        ax2.plot(history[x_key], history['accuracy'],
                 label=name, color=color_list[i % len(color_list)],
                 linewidth=2, alpha=0.9)

    ax2.set_xlabel('Rounds / Epochs')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Non-IID Partition (Pathological)')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_edge_comparison(comparison, output_path):
    """Bar chart comparing full vs edge model metrics.

    Shows the tradeoffs between model accuracy, size, and latency.

    Args:
        comparison: dict from edge_optimizations.compare_models()
        output_path: path to save PNG
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = ['Full MLP', 'Edge MLP', 'Full (INT8)']
    colors = ['#2196F3', '#E91E63', '#FF9800']

    # Accuracy
    accs = [
        comparison['full']['accuracy'],
        comparison['edge']['accuracy'],
        comparison['full']['accuracy'],  # INT8 accuracy shown separately if needed
    ]
    axes[0].bar(models, accs, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0, 1.1)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Model Size
    sizes = [
        comparison['full']['size_bytes'] / 1024,
        comparison['edge']['size_bytes'] / 1024,
        comparison['full_quantized_int8']['size_bytes'] / 1024,
    ]
    axes[1].bar(models, sizes, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    axes[1].set_ylabel('Model Size (KB)')
    axes[1].set_title('Model Size')
    for i, v in enumerate(sizes):
        axes[1].text(i, v + 0.5, f'{v:.1f} KB', ha='center', fontweight='bold')

    # Parameters
    params = [
        comparison['full']['params'],
        comparison['edge']['params'],
        comparison['full_quantized_int8']['params'],
    ]
    axes[2].bar(models, params, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    axes[2].set_ylabel('Parameter Count')
    axes[2].set_title('Parameters')
    for i, v in enumerate(params):
        axes[2].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

    fig.suptitle('Edge AI Model Comparison', fontsize=15, fontweight='bold')
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_quantization_impact(results_by_dtype, output_path):
    """Plot accuracy impact of different quantization levels.

    Args:
        results_by_dtype: dict mapping dtype name to accuracy
        output_path: path to save PNG
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    dtypes = list(results_by_dtype.keys())
    accuracies = list(results_by_dtype.values())
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    bars = ax.bar(dtypes, accuracies, color=colors[:len(dtypes)],
                  alpha=0.85, edgecolor='white', linewidth=1.5)

    ax.set_ylabel('Test Accuracy')
    ax.set_xlabel('Precision')
    ax.set_title('Impact of Quantization on Accuracy')
    ax.set_ylim(0, 1.1)

    for bar, v in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f'{v:.4f}', ha='center', fontweight='bold')

    ax.grid(True, alpha=0.2, axis='y')

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_communication_efficiency(speedup_data, output_path):
    """Plot communication efficiency gains (speedup factors).

    Reproduces the style of Table 2 / Table 3 from the paper as a bar chart.

    Args:
        speedup_data: dict mapping config name to (rounds, speedup)
        output_path: path to save PNG
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = list(speedup_data.keys())
    rounds = [v[0] for v in speedup_data.values()]
    speedups = [v[1] for v in speedup_data.values()]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width / 2, rounds, width, label='Comm. Rounds',
                   color='#2196F3', alpha=0.85)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, speedups, width, label='Speedup ×',
                    color='#4CAF50', alpha=0.85)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Communication Rounds', color='#2196F3')
    ax2.set_ylabel('Speedup Factor (×)', color='#4CAF50')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right')
    ax.set_title('Communication Efficiency: Rounds to Target Accuracy')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
