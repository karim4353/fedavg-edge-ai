"""End-to-end smoke test: run the full driver on a tiny config."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.run_all import main


def test_smoke_run_produces_all_artifacts(tmp_path: Path):
    cfg = {
        "seed": 0,
        "n_samples": 200,
        "n_features": 8,
        "n_classes": 4,
        "n_hidden": 8,
        "n_clients": 3,
        "rounds": 2,
        "lr": 0.05,
        "batch_size": 16,
        "fedavg_local_epochs": 1,
        "client_fraction": 1.0,
        "out_dir": str(tmp_path / "reports"),
    }
    cfg_path = tmp_path / "smoke.json"
    cfg_path.write_text(json.dumps(cfg))

    rc = main(["--config", str(cfg_path)])
    assert rc == 0

    out = Path(cfg["out_dir"])
    # Tables
    assert (out / "tables" / "results.csv").exists()
    assert (out / "tables" / "edge_ai.csv").exists()
    # Plots
    assert (out / "plots" / "convergence.png").exists()
    assert (out / "plots" / "iid_vs_noniid.png").exists()
    assert (out / "plots" / "edge_ai.png").exists()
    # Report
    assert (out / "report.md").exists()


def test_smoke_results_have_expected_experiments(tmp_path: Path):
    cfg = {
        "seed": 0,
        "n_samples": 200,
        "n_features": 8,
        "n_classes": 4,
        "n_hidden": 8,
        "n_clients": 3,
        "rounds": 2,
        "lr": 0.05,
        "batch_size": 16,
        "fedavg_local_epochs": 1,
        "client_fraction": 1.0,
        "out_dir": str(tmp_path / "reports"),
    }
    cfg_path = tmp_path / "smoke.json"
    cfg_path.write_text(json.dumps(cfg))
    assert main(["--config", str(cfg_path)]) == 0

    df = pd.read_csv(Path(cfg["out_dir"]) / "tables" / "results.csv")
    expected = {"centralized", "fedsgd_iid", "fedavg_iid", "fedavg_noniid"}
    assert expected.issubset(set(df["experiment"].tolist()))

    edge = pd.read_csv(Path(cfg["out_dir"]) / "tables" / "edge_ai.csv")
    edge_expected = {"fedavg_baseline", "quantized_int8",
                     "pruned_50", "pruned_70", "compact_mlp"}
    assert edge_expected.issubset(set(edge["experiment"].tolist()))
