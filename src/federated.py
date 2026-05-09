"""Federated training orchestrator.

`run_federated` implements the inner loop common to FedSGD and FedAvg
(McMahan et al. 2017, Algorithm 1). Specializing the local update knobs:

    FedSGD : local_epochs=1,  batch_size=None (full local batch)
    FedAvg : local_epochs>=1, batch_size>=1  (typically B=10..50)
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .client import local_train
from .model import MLP
from .server import weighted_average

EvalFn = Callable[[MLP], dict]


def run_federated(
    model: MLP,
    client_data: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_rounds: int,
    lr: float,
    local_epochs: int,
    batch_size: Optional[int],
    client_fraction: float = 1.0,
    seed: int = 42,
    eval_fn: Optional[EvalFn] = None,
) -> List[dict]:
    """Run federated training. Mutates `model` in place; returns history."""
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    if not (0.0 < client_fraction <= 1.0):
        raise ValueError("client_fraction must be in (0, 1]")

    rng = np.random.default_rng(seed)
    history: List[dict] = []
    n_clients = len(client_data)
    if n_clients == 0:
        raise ValueError("no clients")

    for r in range(n_rounds):
        m = max(1, int(round(client_fraction * n_clients)))
        selected = rng.choice(n_clients, size=m, replace=False)

        global_params = model.get_params()
        client_params: list[list[np.ndarray]] = []
        client_weights: list[int] = []

        for c_idx in selected:
            Xc, yc = client_data[int(c_idx)]
            local_model = model.clone()
            local_model.set_params(global_params)
            B = batch_size if batch_size is not None else max(1, len(yc))
            params = local_train(
                local_model,
                Xc,
                yc,
                lr=lr,
                epochs=local_epochs,
                batch_size=B,
                seed=seed + r * 1000 + int(c_idx),
            )
            client_params.append(params)
            client_weights.append(int(len(yc)))

        new_global = weighted_average(client_params, client_weights)
        model.set_params(new_global)

        if eval_fn is not None:
            metrics = eval_fn(model)
            metrics["round"] = r + 1
            history.append(metrics)

    return history
