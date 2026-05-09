"""End-to-end experiment driver.

Runs the full Student-5 experiment matrix:

    1. Centralized baseline               (upper-bound reference)
    2. FedSGD on IID data                 (paper baseline)
    3. FedAvg on IID data                 (paper proposed)
    4. FedAvg on pathological non-IID     (paper Section 3 stress test)
    5. Edge AI variants on (3):
         - INT8 quantization               (post-training)
         - Magnitude pruning at 50% & 70%  (post-training)
         - Compact-width MLP (FedAvg)      (trained from scratch)

Outputs (under `reports/`):
    tables/results.csv            - one row per experiment
    tables/edge_ai.csv            - edge-AI specific metrics + size
    plots/convergence.png         - accuracy-vs-round for all FL variants
    plots/edge_ai.png             - accuracy / size trade-off
    plots/iid_vs_noniid.png       - paper Section 3 reproduction
    report.md                     - short markdown summary

Configurable via a JSON file under `configs/`. Default config is small but
informative; the smoke config is calibrated for CI.

Usage:
    python -m experiments.run_all --config configs/default.json
    python -m experiments.run_all --config configs/smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.client import local_train  # noqa: E402
from src.data import (  # noqa: E402
    make_synthetic_classification,
    partition_iid,
    partition_noniid_by_label,
    train_test_split,
)
from src.edge import (  # noqa: E402
    compact_mlp,
    magnitude_prune,
    model_sparsity,
    pruned_size_bytes,
    quantize_int8,
    quantized_size_bytes,
)
from src.federated import run_federated  # noqa: E402
from src.metrics import accuracy, cross_entropy, macro_f1  # noqa: E402
from src.model import MLP  # noqa: E402
from src.utils import ensure_dir, set_seed, stopwatch  # noqa: E402


# --------------------------------------------------------------------- config
DEFAULT_CONFIG: Dict = {
    "seed": 42,
    "n_samples": 2000,
    "n_features": 32,
    "n_classes": 5,
    "n_hidden": 32,
    "n_clients": 10,
    "rounds": 30,
    "lr": 0.05,
    "batch_size": 32,
    "fedavg_local_epochs": 3,
    "client_fraction": 1.0,
    "out_dir": "reports",
}


def load_config(path: str | None) -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        cfg.update(user)
    return cfg


# --------------------------------------------------------------------- helpers
def make_eval_fn(X_test: np.ndarray, y_test: np.ndarray, n_classes: int):
    def _eval(model: MLP) -> dict:
        P, _ = model.forward(X_test)
        y_pred = P.argmax(axis=1)
        return {
            "accuracy": accuracy(y_test, y_pred),
            "macro_f1": macro_f1(y_test, y_pred, n_classes),
            "loss": cross_entropy(P, y_test),
        }

    return _eval


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    return make_eval_fn(X, y, n_classes)(model)


def centralized_train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    lr: float,
    rounds: int,
    batch_size: int,
    seed: int,
) -> List[dict]:
    """Round-equivalent centralized training: one full pass per round."""
    eval_fn = make_eval_fn(X_test, y_test, n_classes)
    history = []
    rng = np.random.default_rng(seed)
    n = len(y)
    for r in range(rounds):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s : s + batch_size]
            _, cache = model.forward(X[b])
            grads = model.backward(cache, y[b])
            model.step(grads, lr)
        m = eval_fn(model)
        m["round"] = r + 1
        history.append(m)
    return history


# ---------------------------------------------------------------------- plots
def plot_convergence(histories: Dict[str, List[dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        rounds = [h["round"] for h in hist]
        accs = [h["accuracy"] for h in hist]
        ax.plot(rounds, accs, label=name, linewidth=1.8)
    ax.set_xlabel("Communication round (= epoch for centralized)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Convergence: centralized vs FedSGD vs FedAvg (IID & non-IID)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_iid_vs_noniid(
    iid_hist: List[dict], noniid_hist: List[dict], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([h["round"] for h in iid_hist], [h["accuracy"] for h in iid_hist],
            label="FedAvg — IID", linewidth=2, color="tab:blue")
    ax.plot([h["round"] for h in noniid_hist], [h["accuracy"] for h in noniid_hist],
            label="FedAvg — pathological non-IID", linewidth=2, color="tab:red",
            linestyle="--")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Effect of data heterogeneity on FedAvg (paper §3)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_edge_ai(rows: List[dict], out_path: Path) -> None:
    names = [r["experiment"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    sizes_kb = [r["size_bytes"] / 1024.0 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(names, accs, color="tab:blue")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Accuracy of edge variants")
    axes[0].set_ylim(0.0, 1.0)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, sizes_kb, color="tab:orange")
    axes[1].set_ylabel("On-device size (kB)")
    axes[1].set_title("Storage footprint")
    for i, v in enumerate(sizes_kb):
        axes[1].text(i, v + max(sizes_kb) * 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Edge AI: accuracy vs storage trade-off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------- driver
def _df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table without `tabulate`."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, float_fmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_report(cfg: Dict, results_df: pd.DataFrame, edge_df: pd.DataFrame,
                 out_path: Path) -> None:
    lines = []
    lines.append("# Experiment report — Student 5 contribution\n")
    lines.append(
        "_Generated automatically by `experiments/run_all.py`. "
        "All runs are deterministic for the given seed._\n"
    )
    lines.append("## Configuration\n")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2))
    lines.append("```\n")

    lines.append("## Federated vs centralized (final-round metrics)\n")
    lines.append(_df_to_markdown(results_df))
    lines.append("")

    lines.append("## Edge AI variants\n")
    lines.append(_df_to_markdown(edge_df))
    lines.append("")

    lines.append("## Figures\n")
    lines.append("- `plots/convergence.png`: per-round accuracy of all FL variants.")
    lines.append("- `plots/iid_vs_noniid.png`: data-heterogeneity stress test.")
    lines.append("- `plots/edge_ai.png`: accuracy/size trade-off across edge variants.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FedAvg + Edge AI experiment driver")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config file. Defaults to in-script defaults.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    out_dir = ensure_dir(cfg["out_dir"])
    plots_dir = ensure_dir(out_dir / "plots")
    tables_dir = ensure_dir(out_dir / "tables")

    # ---------- data ----------
    X, y = make_synthetic_classification(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        n_classes=cfg["n_classes"],
        class_imbalance=True,
        seed=cfg["seed"],
    )
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=0.2, seed=cfg["seed"])
    iid_clients = partition_iid(X_train, y_train, cfg["n_clients"], seed=cfg["seed"])
    noniid_clients = partition_noniid_by_label(
        X_train, y_train, cfg["n_clients"], shards_per_client=2, seed=cfg["seed"]
    )
    eval_fn = make_eval_fn(X_test, y_test, cfg["n_classes"])

    histories: Dict[str, List[dict]] = {}
    rows: List[dict] = []

    # ---------- 1. centralized ----------
    print("[1/5] Centralized baseline ...", flush=True)
    model = MLP(cfg["n_features"], cfg["n_hidden"], cfg["n_classes"], seed=cfg["seed"])
    with stopwatch() as sw:
        hist = centralized_train(
            model, X_train, y_train, X_test, y_test,
            n_classes=cfg["n_classes"], lr=cfg["lr"], rounds=cfg["rounds"],
            batch_size=cfg["batch_size"], seed=cfg["seed"],
        )
    histories["centralized"] = hist
    rows.append({
        "experiment": "centralized", **hist[-1],
        "time_s": round(sw["elapsed"], 3),
        "size_bytes": model.size_bytes(),
        "n_params": model.num_params(),
    })

    # ---------- 2. FedSGD (IID) ----------
    print("[2/5] FedSGD (IID) ...", flush=True)
    model = MLP(cfg["n_features"], cfg["n_hidden"], cfg["n_classes"], seed=cfg["seed"])
    with stopwatch() as sw:
        hist = run_federated(
            model, iid_clients, n_rounds=cfg["rounds"],
            lr=cfg["lr"], local_epochs=1, batch_size=None,
            client_fraction=cfg["client_fraction"], seed=cfg["seed"], eval_fn=eval_fn,
        )
    histories["fedsgd_iid"] = hist
    rows.append({
        "experiment": "fedsgd_iid", **hist[-1],
        "time_s": round(sw["elapsed"], 3),
        "size_bytes": model.size_bytes(),
        "n_params": model.num_params(),
    })

    # ---------- 3. FedAvg (IID) ----------
    print("[3/5] FedAvg (IID) ...", flush=True)
    model_iid = MLP(cfg["n_features"], cfg["n_hidden"], cfg["n_classes"], seed=cfg["seed"])
    with stopwatch() as sw:
        hist = run_federated(
            model_iid, iid_clients, n_rounds=cfg["rounds"],
            lr=cfg["lr"], local_epochs=cfg["fedavg_local_epochs"],
            batch_size=cfg["batch_size"],
            client_fraction=cfg["client_fraction"], seed=cfg["seed"], eval_fn=eval_fn,
        )
    histories["fedavg_iid"] = hist
    rows.append({
        "experiment": "fedavg_iid", **hist[-1],
        "time_s": round(sw["elapsed"], 3),
        "size_bytes": model_iid.size_bytes(),
        "n_params": model_iid.num_params(),
    })

    # ---------- 4. FedAvg (non-IID) ----------
    print("[4/5] FedAvg (non-IID) ...", flush=True)
    model_noniid = MLP(cfg["n_features"], cfg["n_hidden"], cfg["n_classes"], seed=cfg["seed"])
    with stopwatch() as sw:
        hist = run_federated(
            model_noniid, noniid_clients, n_rounds=cfg["rounds"],
            lr=cfg["lr"], local_epochs=cfg["fedavg_local_epochs"],
            batch_size=cfg["batch_size"],
            client_fraction=cfg["client_fraction"], seed=cfg["seed"], eval_fn=eval_fn,
        )
    histories["fedavg_noniid"] = hist
    rows.append({
        "experiment": "fedavg_noniid", **hist[-1],
        "time_s": round(sw["elapsed"], 3),
        "size_bytes": model_noniid.size_bytes(),
        "n_params": model_noniid.num_params(),
    })

    # ---------- 5. Edge AI variants (built off the IID FedAvg model) ----------
    print("[5/5] Edge AI variants ...", flush=True)
    edge_rows: List[dict] = []

    base_metrics = evaluate(model_iid, X_test, y_test, cfg["n_classes"])
    edge_rows.append({
        "experiment": "fedavg_baseline",
        **base_metrics,
        "size_bytes": model_iid.size_bytes(),
        "sparsity": 0.0,
        "n_params": model_iid.num_params(),
    })

    # 5a. INT8 quantization
    qmodel = quantize_int8(model_iid)
    qm = evaluate(qmodel, X_test, y_test, cfg["n_classes"])
    edge_rows.append({
        "experiment": "quantized_int8",
        **qm,
        "size_bytes": quantized_size_bytes(qmodel),
        "sparsity": 0.0,
        "n_params": qmodel.num_params(),
    })

    # 5b. Magnitude pruning at two sparsity levels
    for sp in (0.5, 0.7):
        pmodel = magnitude_prune(model_iid, sparsity=sp)
        pm = evaluate(pmodel, X_test, y_test, cfg["n_classes"])
        edge_rows.append({
            "experiment": f"pruned_{int(sp*100)}",
            **pm,
            "size_bytes": pruned_size_bytes(pmodel),
            "sparsity": model_sparsity(pmodel),
            "n_params": pmodel.num_params(),
        })

    # 5c. Compact-width FedAvg (trained from scratch)
    small = compact_mlp(model_iid, factor=2, seed=cfg["seed"])
    with stopwatch():
        run_federated(
            small, iid_clients, n_rounds=cfg["rounds"],
            lr=cfg["lr"], local_epochs=cfg["fedavg_local_epochs"],
            batch_size=cfg["batch_size"],
            client_fraction=cfg["client_fraction"], seed=cfg["seed"], eval_fn=eval_fn,
        )
    sm = evaluate(small, X_test, y_test, cfg["n_classes"])
    edge_rows.append({
        "experiment": "compact_mlp",
        **sm,
        "size_bytes": small.size_bytes(),
        "sparsity": 0.0,
        "n_params": small.num_params(),
    })

    # ---------- write artifacts ----------
    results_df = pd.DataFrame(rows)
    edge_df = pd.DataFrame(edge_rows)

    results_csv = tables_dir / "results.csv"
    edge_csv = tables_dir / "edge_ai.csv"
    results_df.to_csv(results_csv, index=False)
    edge_df.to_csv(edge_csv, index=False)

    plot_convergence(histories, plots_dir / "convergence.png")
    plot_iid_vs_noniid(histories["fedavg_iid"], histories["fedavg_noniid"],
                       plots_dir / "iid_vs_noniid.png")
    plot_edge_ai(edge_rows, plots_dir / "edge_ai.png")

    write_report(cfg, results_df, edge_df, out_dir / "report.md")

    print("\n=== Final results ===")
    print(results_df.to_string(index=False))
    print("\n=== Edge AI variants ===")
    print(edge_df.to_string(index=False))
    print(f"\nArtifacts written under: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
