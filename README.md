# FedAvg + Edge AI — Experimental Reproduction

[![CI](https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg)](../../actions)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A lightweight, deterministic, CPU-only reproduction of the **FedAvg** algorithm
from McMahan et al. 2017 — *Communication-Efficient Learning of Deep Networks
from Decentralized Data* — extended with a small set of **Edge AI**
optimizations (post-training INT8 quantization, magnitude pruning, compact-width
model variant).

This repository implements **Student 5's part** of the team project
*“Federated learning for IoT devices — Enhancing TinyML with on-board training”*:
the experimental simulation, results, and embedded-AI improvements section.

---

## Project purpose

The team's overall project investigates Federated Learning (FL) on resource-
constrained IoT devices, combining FL with Transfer Learning over MQTT, with
hardware ranging from Arduino WiFi Rev2 to ESP32 to Raspberry Pi.

**This repository covers only the simulation/experimentation side**:

* a clean, reproducible NumPy implementation of FedSGD and FedAvg,
* IID vs. pathological non-IID partitioning (paper §3),
* a centralized baseline,
* three realistic Edge AI improvements applied to the federated model,
* a comparison table + plots produced automatically.

Everything runs on CPU in well under a minute on a laptop, and is exercised in
GitHub Actions on every push.

## Relation to the paper

| Paper element                                 | This repo                                                      | Notes                                                                |
| --------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------- |
| FederatedAveraging (Algorithm 1)              | `src/federated.py` + `src/server.weighted_average`             | Full faithful implementation                                         |
| FedSGD baseline                               | `run_federated(local_epochs=1, batch_size=None)`               | Special case of FedAvg, as in the paper                              |
| IID partition                                 | `src/data.partition_iid`                                       | Shuffle + split                                                      |
| Pathological non-IID (sort-by-label, 2-shards) | `src/data.partition_noniid_by_label`                           | Same scheme as paper §3                                              |
| MNIST 2NN / CNN, CIFAR, Shakespeare LSTM      | **Not reproduced**                                             | Outside the resource budget for offline + CI runs                    |
| Synthetic ECG-like dataset                    | `src/data.make_synthetic_classification`                       | 5-class, ~71/16/6/5/2 % imbalance, mimicking the team's ECG dataset  |
| Communication-rounds metric                   | `reports/tables/results.csv` + `plots/convergence.png`         | Per-round accuracy curves                                            |

The **algorithmic** contributions (FedSGD, FedAvg, IID/non-IID partitioning,
weighted aggregation) are reproduced exactly. The **datasets and architectures**
are simplified to a small synthetic problem so that the entire experiment
matrix runs offline in seconds — required for GitHub Actions and for honest,
reproducible reporting on a laptop.

## Student 5 contribution

* Designed and implemented the **federated simulation framework** from scratch
  in pure NumPy:
  * 2-layer MLP (forward/backward/SGD) — `src/model.py`
  * client-side local training — `src/client.py`
  * server-side weighted aggregation — `src/server.py`
  * round-by-round orchestrator — `src/federated.py`
* Implemented and validated **IID vs. pathological non-IID partitioning**
  matching the paper §3.
* Designed the synthetic, ECG-inspired benchmark dataset so the project can run
  offline and on CI with no proprietary data.
* Wrote three **Edge AI optimizations** on top of the trained federated model
  (`src/edge.py`); see next section.
* Produced the experiment matrix, CSV result tables, plots, and Markdown
  report (`experiments/run_all.py`).
* Wrote the unit + integration test suite (`tests/`) — aggregation correctness,
  partitioning behavior, determinism under fixed seeds, end-to-end smoke run.
* Set up the **GitHub Actions CI/CD pipeline** that installs deps, runs tests,
  runs the smoke experiment, verifies outputs, and uploads the report as an
  artifact (`.github/workflows/ci.yml`).

## Edge AI improvement

The team's presentation identifies the core deployment constraints — RAM in
the kB range, weak CPUs, and the cost of using `double` instead of `float` on
microcontrollers. Three resource-aware improvements are implemented, all
reusing the same federated model:

1. **Post-training INT8 weight quantization** (per-tensor, symmetric) —
   `src.edge.quantize_int8`.
   Reduces storage from float32 (4 B/weight) to int8 (1 B/weight) plus one
   fp32 scale per tensor — close to a **4× model-size reduction**, with a
   typically negligible accuracy drop on this small MLP.

2. **Magnitude pruning** at 50 % and 70 % sparsity — `src.edge.magnitude_prune`.
   Zeros the smallest-magnitude weights of `W1` and `W2` and reports a
   realistic CSR-style on-device storage estimate.

3. **Compact-width MLP** (`hidden = hidden / 2`) — `src.edge.compact_mlp`.
   Re-trained from scratch with FedAvg under the same client distribution,
   isolating the cost of width vs. compression.

Each variant is benchmarked on the *same* held-out test set and reported in
`reports/tables/edge_ai.csv` and `reports/plots/edge_ai.png` (accuracy and
on-device size, side by side).

## How to install

```bash
git clone https://github.com/USER/REPO.git
cd REPO
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

CPU-only, Python 3.11. No GPU, no PyTorch, no TensorFlow.

## How to run experiments

```bash
# Default ~ 30 rounds, 10 clients, 2 000 samples (a few seconds on a laptop)
python -m experiments.run_all --config configs/default.json

# Tiny smoke run (used by CI; under 5 seconds)
python -m experiments.run_all --config configs/smoke.json
```

Or via the Makefile:

```bash
make run     # full default config
make smoke   # tiny CI-friendly config
```

## How to run tests

```bash
pytest -q
```

Or `make test`.

## How to reproduce results

All randomness is seeded (`src.utils.set_seed`); the same config and seed
produce identical numbers across machines. To reproduce the headline numbers:

```bash
python -m experiments.run_all --config configs/default.json
cat reports/tables/results.csv
cat reports/tables/edge_ai.csv
```

## Expected outputs

After a run, `reports/` contains:

```
reports/
├── tables/
│   ├── results.csv          # one row per FL/centralized experiment
│   └── edge_ai.csv          # one row per edge variant (size + accuracy)
├── plots/
│   ├── convergence.png      # per-round accuracy of all FL variants
│   ├── iid_vs_noniid.png    # paper §3 stress test
│   └── edge_ai.png          # accuracy / size trade-off bar charts
└── report.md                # auto-generated markdown summary
```

The CI pipeline uploads all of `reports/` as an artifact named
`smoke-reports-3.11` on every run.

## Repository layout

```
fedavg-edge-ai/
├── src/                      Library code
│   ├── data.py               Synthetic dataset + IID/non-IID partitioning
│   ├── model.py              NumPy MLP (forward/backward/SGD)
│   ├── client.py             Local SGD training
│   ├── server.py             Weighted averaging
│   ├── federated.py          FedSGD / FedAvg orchestrator
│   ├── edge.py               Quantization, pruning, compact-width
│   ├── metrics.py            Accuracy, macro-F1, cross-entropy
│   └── utils.py              Seeding + IO helpers
├── experiments/
│   └── run_all.py            End-to-end experiment driver
├── configs/
│   ├── default.json
│   └── smoke.json
├── tests/                    pytest suite
├── reports/                  Generated artifacts (git-ignored)
├── .github/workflows/ci.yml  GitHub Actions
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## Limitations and future work

* **Synthetic dataset.** The dataset mimics the *qualitative* properties of
  the ECG heartbeat task (5 imbalanced classes, compact features) but is not
  the real PhysioNet/MIT-BIH data. Final accuracy numbers should not be
  compared directly to the paper's MNIST/CIFAR/Shakespeare numbers — the
  comparisons that matter here are the *relative* ones (FedAvg vs. FedSGD;
  IID vs. non-IID; baseline vs. edge variants). A future version could plug
  in a downloaded ECG snapshot when run interactively, while keeping the
  synthetic fallback for CI.
* **No Transfer Learning leg.** The team's proposed solution combines FL with
  TL on a server-pretrained model. This repo focuses on the FL/Edge axis;
  a TL leg (server pretrains on shard 0, clients fine-tune) would be a
  natural extension and would let us reproduce the FL+TL row of the team's
  ECG results table.
* **No regression task.** The team also runs a Car Trips Data Log regression
  experiment; only the classification arm is implemented here.
* **No real MQTT layer.** The federated round is a function call, not a
  network exchange. A future iteration could add a thread-based MQTT mock to
  match the team's deployment architecture more faithfully.
* **Quantization is post-training.** A quantization-aware training (QAT)
  variant — even a simple straight-through estimator — would likely close the
  small accuracy gap observed for INT8.
* **Single seed reported.** The CSVs report a single deterministic run per
  config. A multi-seed sweep with mean ± std would harden the conclusions and
  is a one-line addition to `run_all.py`.

## License

MIT.
