<div align="center">

# Deep Learning Benchmark Suite

**A reproducible PyTorch project for benchmarking neural architectures across image classification, specialized deep learning tasks, and multimodal learning.**

[![Project Status](https://img.shields.io/badge/status-active%20development-2563eb)](https://github.com/TuanKhai1210/Deep-Learning-Benchmark-Suite)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Project Site](https://img.shields.io/badge/GitHub%20Pages-project%20site-222222?logo=github)](https://tuankhai1210.github.io/Deep-Learning-Benchmark-Suite/)

[Project website](https://tuankhai1210.github.io/Deep-Learning-Benchmark-Suite/) ·
[Assignment 1](docs/a1.md) ·
[AI usage disclosure](AI_USAGE.md)

</div>

---

## Overview

Deep Learning Benchmark Suite is an evolving collection of controlled experiments designed to answer a practical question:

> How do different deep learning architectures behave when they are trained and evaluated under the same data split, metrics, and reproducibility protocol?

The project begins with image classification and progressively extends toward specialized and multimodal learning. It emphasizes reproducible engineering, fair model comparison, transparent experiment tracking, and evidence-based analysis rather than accuracy alone.

## Project modules

| Module | Focus | Dataset or task | Status |
|---|---|---|---|
| Assignment 1 | Architecture benchmarking and end-to-end training pipelines | Fashion-MNIST image classification | In progress |
| Assignment 2 | Large-scale data and a specialized vision or language task | Pending dataset proposal and approval | Planned |
| Assignment 3 | Multimodal representation, fusion, and evaluation | Pending dataset proposal and approval | Planned |

The current development priority is **Assignment 1**.

Configuration validation and the command-line interface are available. Data preparation, model training, evaluation, and analysis are under development; no benchmark results are reported yet. See the [implementation guide](IMPLEMENTATION_GUIDE.md) for file ownership and implementation order.

## Assignment 1: Architecture Benchmark

Assignment 1 compares five model families on the same Fashion-MNIST split and evaluation protocol:

1. Linear classifier
2. Multilayer Perceptron (MLP)
3. Custom Convolutional Neural Network (CNN)
4. LSTM or GRU sequence model
5. Transformer encoder

### Evaluation dimensions

- Accuracy and macro-F1
- Parameter count
- Training and inference time
- Training and validation curves
- Confusion matrix
- Correct, difficult, and incorrect prediction examples
- Representation and inductive-bias analysis

### Reproducibility goals

Every reported result should be traceable to:

- A versioned configuration
- A fixed dataset split and random seed
- An experiment log or identifier
- A selected checkpoint
- The hardware and dependency versions used
- A corresponding Git commit or release tag

The team has agreed on four training seeds: **36, 69420, 67, and 69**. The planned final comparison has five model families × four seeds = **20 main training runs**, excluding debugging and tuning. All runs use one saved split; the separate split seed of `42` remains a proposal. Final results will report mean and standard deviation across all four seeds, without selecting the best test seed.

Other experimental choices are still under review in the [A1 experiment contract](docs/a1-experiment-contract.md), including normalization statistics, compute budget, augmentation, and the final timing procedure.

## Repository structure

```text
deep-learning-benchmark-suite/
├── README.md                  # Project overview and setup
├── IMPLEMENTATION_GUIDE.md     # File ownership and development order
├── CONTRIBUTING.md            # Branches, reviews, and verification
├── PROJECT_PLAN.md            # Milestones and team responsibilities
├── AI_USAGE.md                # Verifiable AI-assistance log
├── setup.py                   # Python package and dependency declarations
├── configs/
│   ├── a1/
│   │   ├── protocol.py        # Shared experiment policy
│   │   ├── models/            # One Python config per model family
│   │   └── splits/            # Versioned split indices and identity
│   ├── a2/
│   └── a3/
├── src/dlbench/
│   ├── common/                # Config, reproducibility, and artifacts
│   └── a1/
│       ├── data/              # Dataset, split, transforms, loaders, EDA
│       ├── models/            # Linear, MLP, CNN, RNN, Transformer
│       ├── contracts.py       # Shared batch/model/metric interfaces
│       ├── cli.py             # Command-line entry point
│       ├── engine.py          # Epoch-level training and evaluation
│       └── trainer.py         # Full-run orchestration
├── tests/                     # Configuration and implementation tests
├── environment/               # Dependency and hardware records
├── notebooks/                 # Exploratory analysis, not the main pipeline
├── docs/                      # GitHub Pages and experiment contract
│   └── assets/a1/             # Small reviewed tables and figures
├── reports/a1/                # Report and presentation outline
├── data/                      # Local datasets, excluded from Git
├── runs/                      # Local run artifacts, excluded from Git
└── checkpoints/               # Model weights, excluded from Git
```

Datasets, checkpoints, local experiment logs, and credentials are intentionally excluded from Git. Checkpoint download links or reconstruction instructions will be documented with each release.

## Getting started

Clone the repository:

```bash
git clone https://github.com/TuanKhai1210/deep-learning-benchmark-suite.git
cd deep-learning-benchmark-suite
```

Use Python **3.11 or newer**. Create an isolated environment, then install the local package. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m dlbench.a1.cli --help
.\.venv\Scripts\python.exe -m dlbench.a1.cli validate-config --config configs/a1/models/linear.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This base installation supports development checks without installing the ML stack. After choosing a compatible PyTorch/torchvision build for the team's hardware, install the optional ML dependencies and record the resolved versions:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
```

Experiment configs are `.py` files containing a single literal `CONFIG = {...}` dictionary. The shared rules live in `configs/a1/protocol.py`; model-specific settings live in `configs/a1/models/`. The loader parses data without importing or executing the file. Comments/docstrings are allowed; imports, calls, variables and computations are not.

`validate-config` checks development settings. Add `--strict` only after the team has genuinely completed and approved the protocol; unresolved values should fail strict validation. Use `--seed 36` to select one of the four agreed run seeds in memory; it does not change the config file or split. Other seeds are rejected.

The `prepare`, `train`, `evaluate`, and `analyze` commands define the intended workflow but currently stop at unimplemented ML functions. They are not evidence of a completed training pipeline. ML-dependent tests remain pending until their implementations and dependencies are ready.

Follow progress on the [Assignment 1 project page](docs/a1.md).

## Development workflow

The team uses short-lived feature branches and pull requests:

```text
feat/a1-data-eda
feat/a1-training-pipeline
feat/a1-models
docs/a1-results
```

Contributors should:

1. Branch from the latest `main`.
2. Keep commits focused and descriptive.
3. Open a pull request instead of pushing directly to `main`.
4. Request review from at least one teammate.
5. Merge only after the relevant smoke tests and documentation checks pass.

See [CONTRIBUTING.md](CONTRIBUTING.md) for review responsibilities and the pre-merge checklist.

## Roadmap

- [x] Initialize the repository and project documentation
- [x] Publish and verify the GitHub Pages landing page
- [ ] Complete Fashion-MNIST EDA and freeze the shared split
- [ ] Implement the Dataset/DataLoader and train-validation-test pipeline
- [ ] Train reproducible Linear and MLP baselines
- [ ] Complete the Assignment 1 draft milestone
- [ ] Implement CNN, LSTM/GRU, and Transformer models
- [ ] Run the full controlled comparison and error analysis
- [ ] Publish the Assignment 1 report, slides, video, and checkpoints
- [ ] Begin Assignment 2 only after dataset proposal approval

## Team

| Member | Student ID | Primary responsibility | GitHub |
|---|---|---|---|
| Nguyễn Hạo Thiên (A) | 2453194 | Data, EDA, preprocessing; Linear and CNN | [MrzThien1105](https://github.com/MrzThien1105) |
| Nguyễn Anh Khoa (B) | 2452539 | Trainer, evaluation, metrics, timing; MLP and LSTM/GRU | [TCL03-HCMUT](https://github.com/TCL03-HCMUT) |
| Tạ Tuấn Khải (C) | 2452515 | Config, reproducibility, integration/tests; Transformer and report/Pages integration | [TuanKhai1210](https://github.com/TuanKhai1210) |

Review rotation: Khoa reviews Thiên, Khải reviews Khoa, and Thiên reviews Khải. Each member writes and verifies the methodology, results, and analysis for their own work; report integration is not sole authorship.

## AI usage and research integrity

AI-assisted work is disclosed in [`AI_USAGE.md`](AI_USAGE.md). Every AI-generated suggestion must be reviewed, edited when necessary, and verified through source code, real experiment runs, official documentation, or scholarly sources.

The project does not report fabricated data, results, citations, or experiments that were not actually run.

## Academic context

This repository is developed as part of **CO3133 - Deep Learning and Its Applications**, Semester 261, at the Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (VNU-HCM).

- Instructor: Lê Thành Sách
- Team size: 3 students
- Project website: [Deep Learning Benchmark Suite](https://tuankhai1210.github.io/Deep-Learning-Benchmark-Suite/)

Course provenance is retained for transparency, while the repository is structured as a maintainable and reproducible machine-learning portfolio project.
