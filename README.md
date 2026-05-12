# FedAvg Reproduction & Edge AI Optimization

> **Reproduction of "Communication-Efficient Learning of Deep Networks from Decentralized Data"**
> (McMahan et al., AISTATS 2017) with Edge AI improvements for IoT deployment.

[![CI](https://github.com/YOUR_USERNAME/fedavg-edge-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/fedavg-edge-ai/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![CPU Only](https://img.shields.io/badge/device-CPU%20only-green.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Project Purpose

This repository is the **experimental companion** for our team presentation on:
**"Federated Learning for IoT Devices — Enhancing TinyML with On-Board Training"**

It reproduces the core ideas of the FedAvg paper in a lightweight, deterministic simulation
and extends them with practical Edge AI optimizations suitable for resource-constrained IoT devices.

### Relation to the Paper

The original paper by McMahan et al. (2017) introduced **Federated Averaging (FedAvg)**, an algorithm
that reduces the communication rounds needed to train deep networks by performing multiple local SGD
steps on each client before averaging. Key findings reproduced here:

| Paper Claim | Our Reproduction |
|-------------|-----------------|
| FedAvg reduces communication rounds vs FedSGD | ✓ Confirmed on digits dataset |
| Increasing local epochs E speeds convergence | ✓ Confirmed with E=1,5,10 |
| Non-IID data degrades but doesn't break FedAvg | ✓ Confirmed with pathological split |
| FedAvg with E=1, B=inf equals FedSGD | ✓ Verified mathematically in tests |

### What Is Approximated vs. Fully Reproduced

| Aspect | Paper | Our Reproduction |
|--------|-------|-----------------|
| Dataset | MNIST (70K), CIFAR-10, Shakespeare | Scikit-learn digits (1,797) |
| Model | 2NN (200-unit), CNN, LSTM | MLP (128-64 units) |
| Clients | 100-1146 | 5-10 |
| Rounds | 50-6000 | 10-50 |
| Algorithm | FedSGD, FedAvg | ✓ Faithful implementation |
| IID/Non-IID | Shard-based pathological | ✓ Same strategy |
| Aggregation | Weighted average | ✓ Exact formula |

> **Why digits instead of MNIST?** The digits dataset (64 features, 10 classes) is built into
> scikit-learn, requires no downloads, and has the same task structure. This makes the repo
> fully self-contained and suitable for CI/CD without internet access at runtime.

---

## Student 5 Contribution

**Author:** Skander ABID

As Student 5, my contributions to the team presentation are:

1. **Experimental Reproduction** — Implemented FedSGD and FedAvg from scratch in pure NumPy,
   faithfully following Algorithm 1 from the paper
2. **Simulation Results** — Ran systematic experiments comparing centralized, FedSGD, and
   FedAvg training under IID and non-IID data distributions
3. **Edge AI Improvement** — Designed and evaluated optimizations for deploying federated
   learning on resource-constrained IoT devices (see next section)
4. **CI/CD Pipeline** — Built a GitHub Actions workflow for automated testing and validation

---

## Edge AI Improvement

Our team's presentation focuses on deploying FL on IoT devices (ESP32, ESP8266, Arduino).
These devices have severe constraints:

| Constraint | ESP32 | ESP8266 | Arduino MKR1010 |
|-----------|-------|---------|-----------------|
| RAM | 520 KB | 80 KB | 32 KB |
| CPU | 240 MHz | 80 MHz | 48 MHz |
| Flash | 4 MB | 4 MB | 256 KB |

To address these constraints, I implemented three optimizations:

### 1. Compact Model Architecture
- **Full MLP:** 64 → 128 → 64 → 10 (~17K parameters, ~134 KB)
- **Edge MLP:** 64 → 48 → 24 → 10 (~4.5K parameters, ~35 KB)
- **Result:** ~74% parameter reduction with moderate accuracy loss

### 2. Post-Training Quantization
- float64 → float32 (2× compression)
- float64 → float16 (4× compression)
- float64 → **int8** (8× compression, industry standard for TinyML)
- Combined with Edge MLP: total ~23× size reduction

### 3. Weight Pruning
- Magnitude-based pruning (30-50% sparsity)
- Reduces effective model complexity
- Can be combined with sparse storage for additional memory savings

### Combined Impact

| Configuration | Parameters | Size (KB) | Accuracy | Suitable For |
|---------------|-----------|-----------|----------|-------------|
| Full MLP (float64) | ~17K | 134 | Baseline | PC, Server |
| Full MLP (int8) | ~17K | 17 | ~-0.5% | ESP32 |
| Edge MLP (float32) | ~4.5K | 18 | ~-3% | ESP32, ESP8266 |
| Edge MLP (int8) | ~4.5K | 4.5 | ~-5% | Arduino, ESP8266 |

---

## Installation

### Prerequisites
- Python 3.11
- No GPU required
- No internet access needed at runtime

### Setup

```bash
# Clone the repository
git clone https://github.com/karim4353/fedavg-edge-ai
cd fedavg-edge-ai

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run Experiments

### Quick Smoke Test (~10-30 seconds)
```bash
python experiments/run_smoke.py
```

### Full Experiment Suite (~2-5 minutes)
```bash
python experiments/run_full.py
```

### Using Make (Linux/macOS)
```bash
make install    # Install dependencies
make test       # Run unit tests
make smoke      # Run smoke experiment
make experiment # Run full experiments
make full       # All of the above
```

---

## How to Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_federated.py -v

# Run with short traceback
pytest tests/ -v --tb=short
```

---

## How to Reproduce Results

1. Install dependencies: `pip install -r requirements.txt`
2. Run the full experiment: `python experiments/run_full.py`
3. Check results in `results/full/`:
   - `all_results.csv` — complete metrics table
   - `main_accuracy.png` — accuracy comparison plot
   - `iid_vs_noniid.png` — IID vs non-IID comparison
   - `edge_comparison.png` — full vs edge model comparison
   - `quantization_impact.png` — quantization analysis
   - `communication_efficiency.png` — communication cost analysis
4. Read the generated report: `reports/experiment_report.md`

All experiments use `seed=42` for full deterministic reproducibility.

---

## Expected Outputs

After running `python experiments/run_full.py`:

```
results/full/
├── all_results.csv              # Complete results table
├── quantization_results.csv     # Quantization analysis
├── pruning_results.csv          # Pruning analysis
├── communication_stats.csv      # MQTT communication stats
├── main_accuracy.png            # Main accuracy comparison
├── main_loss.png                # Loss curves
├── iid_vs_noniid.png            # IID vs Non-IID
├── edge_comparison.png          # Full vs Edge model
├── quantization_impact.png      # Quantization impact
└── communication_efficiency.png # Communication efficiency

reports/
└── experiment_report.md         # Auto-generated markdown report
```

---

## Repository Structure

```
fedavg-edge-ai/
├── src/
│   ├── __init__.py
│   ├── data.py                  # Data loading, IID/non-IID partitioning
│   ├── models.py                # MLP models (full + edge)
│   ├── federated.py             # FedSGD, FedAvg, centralized training
│   ├── edge_optimizations.py    # Quantization, pruning, latency
│   ├── mqtt_simulation.py       # MQTT communication mock
│   └── visualization.py         # Plotting utilities
├── experiments/
│   ├── run_smoke.py             # CI-friendly smoke test
│   └── run_full.py              # Complete experiment suite
├── tests/
│   ├── test_data.py             # Data partitioning tests
│   ├── test_models.py           # Model architecture tests
│   ├── test_federated.py        # Aggregation & algorithm tests
│   ├── test_edge.py             # Edge optimization tests
│   └── test_smoke.py            # End-to-end smoke tests
├── configs/
│   ├── smoke.yaml               # Smoke test configuration
│   └── full.yaml                # Full experiment configuration
├── reports/                     # Generated reports
├── results/                     # Generated results (gitignored)
├── .github/workflows/ci.yml    # GitHub Actions CI
├── requirements.txt
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

---

## Limitations and Future Work

### Current Limitations

1. **Simplified dataset:** We use scikit-learn's 64-feature digits dataset instead of the
   full 784-feature MNIST. While the task structure is identical, absolute accuracy numbers
   are not directly comparable to the paper.

2. **Pure NumPy models:** Our MLP implementation uses only NumPy for maximum portability,
   but lacks GPU acceleration and advanced optimizers (Adam, momentum).

3. **No real hardware testing:** Edge AI optimizations are simulated on CPU. Actual
   deployment on ESP32/Arduino would require C/TFLite conversion.

4. **Static quantization:** We use post-training quantization rather than
   quantization-aware training, which could yield better INT8 accuracy.

5. **No differential privacy:** The paper discusses combining FL with differential privacy;
   we do not implement this.

### Future Work

- [ ] Port to actual ESP32 hardware using TensorFlow Lite Micro
- [ ] Implement quantization-aware training for better INT8 accuracy
- [ ] Add differential privacy (DP-SGD) to the aggregation
- [ ] Test with larger datasets (full MNIST, CIFAR-10) via PyTorch
- [ ] Implement asynchronous federated learning for heterogeneous devices
- [ ] Add real MQTT broker integration for multi-device testing
- [ ] Implement model compression techniques (knowledge distillation)
- [ ] Explore FedProx and other FL algorithms for non-IID robustness

---

## References

1. McMahan, H.B., Moore, E., Ramage, D., Hampson, S., & Arcas, B.A.y. (2017).
   *Communication-Efficient Learning of Deep Networks from Decentralized Data.*
   AISTATS 2017. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)

2. Team presentation: *"Federated Learning for IoT Devices — Enhancing TinyML
   with On-Board Training"* (Klidi, Bouziri, Smirani, Sliti, Abid, 2025-2026)

---

## Team

| Member | Role |
|--------|------|
| Sarra KLIDI | Context & Problem Statement |
| Sarra KLIDI | State of the Art |
| Abir BOUZIRI | Proposed Solution |
| Darine SMIRANI | Paper Results Analysis |
| **Skander ABID** | **Experiments, Simulation, Edge AI (this repo)** |
