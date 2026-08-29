<div align="center">

# Deep Learning Benchmark Suite

**A reproducible PyTorch project for benchmarking neural architectures across image classification, specialized deep learning tasks, and multimodal learning.**

[![Project Status](https://img.shields.io/badge/status-active%20development-2563eb)](https://github.com/TuanKhai1210/deep-learning-benchmark-suite)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-framework-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Project Site](https://img.shields.io/badge/GitHub%20Pages-project%20site-222222?logo=github)](https://tuankhai1210.github.io/deep-learning-benchmark-suite/)

[Project website](https://tuankhai1210.github.io/deep-learning-benchmark-suite/) ·
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

## Repository structure

```text
deep-learning-benchmark-suite/
├── README.md                 # Project overview and usage
├── AI_USAGE.md               # Verifiable AI-assistance log
├── docs/                     # GitHub Pages source
│   ├── index.md
│   ├── a1.md
│   ├── a2.md
│   └── a3.md
├── src/
│   └── a1/                   # Assignment 1 implementation
├── configs/
│   └── a1/                   # Versioned experiment configurations
├── tests/                    # Unit and smoke tests
├── results/
│   └── a1/                   # Small, reproducible tables and figures
├── reports/
│   └── a1/
└── slides/
    └── a1/
```

Datasets, checkpoints, local experiment logs, and credentials are intentionally excluded from Git. Checkpoint download links or reconstruction instructions will be documented with each release.

## Getting started

Clone the repository:

```bash
git clone https://github.com/TuanKhai1210/deep-learning-benchmark-suite.git
cd deep-learning-benchmark-suite
```

Installation, dataset preparation, training, and evaluation commands will be added with the first Assignment 1 implementation milestone.

Until then, follow the live progress on the [Assignment 1 project page](docs/a1.md).

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

## Roadmap

- [x] Initialize the repository and documentation skeleton
- [ ] Publish and verify the GitHub Pages landing page
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
| `Tạ Tuấn Khải` | `2452515` | Integration and training pipeline | `TuanKhai1210` |
| `Nguyễn Anh Khoa` | `2452539` | Data, EDA, and evaluation | `TCL03-HCMUT` |
| `[Member 3]` | `[Student ID]` | Models and experiments | `[@username]` |

Replace the placeholders above before publishing the project as a portfolio item.

## AI usage and research integrity

AI-assisted work is disclosed in [`AI_USAGE.md`](AI_USAGE.md). Every AI-generated suggestion must be reviewed, edited when necessary, and verified through source code, real experiment runs, official documentation, or scholarly sources.

The project does not report fabricated data, results, citations, or experiments that were not actually run.

## Academic context

This repository is developed as part of **CO3133 - Deep Learning and Its Applications**, Semester 261, at the Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (VNU-HCM).

- Instructor: Lê Thành Sách
- Team size: 3 students
- Project website: [Deep Learning Benchmark Suite](https://tuankhai1210.github.io/deep-learning-benchmark-suite/)

Course provenance is retained for transparency, while the repository is structured as a maintainable and reproducible machine-learning portfolio project.
