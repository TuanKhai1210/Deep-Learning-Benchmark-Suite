"""Owner A (Thiên): descriptive data analysis, not test-based model tuning."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from dlbench.a1.data.dataset import load_official_dataset
from dlbench.a1.data.split import load_split, validate_split_against_config


def generate_eda(
    config: Mapping[str, Any],
    output_dir: Path,
    curated_dir: Path | None = None,
) -> None:
    """Export counts, class distribution, shape/range checks and representative images.

    Include imbalance and leakage checks, plus dataset source/license.
    Raw preparation outputs go to runs; curated images go to docs/assets/a1.
    """
    data_cfg = config["data"]
    data_root = data_cfg["root"]
    split_path = Path(data_cfg["split_file"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if curated_dir is not None:
        curated_dir = Path(curated_dir)
        curated_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw datasets and split manifest
    raw_train = load_official_dataset(data_root, train=True, download=False)
    raw_test = load_official_dataset(data_root, train=False, download=False)
    manifest = load_split(split_path)
    validate_split_against_config(manifest, data_cfg)

    classes = raw_train.classes
    train_targets = np.array(raw_train.targets)[manifest.train_indices]
    val_targets = np.array(raw_train.targets)[manifest.validation_indices]
    test_targets = np.array(raw_test.targets)[manifest.test_indices]

    train_counts = Counter(train_targets)
    val_counts = Counter(val_targets)
    test_counts = Counter(test_targets)

    # 2. Pixel Shape and Value Range Checks
    first_img, _ = raw_train[0]
    first_tensor = torch.tensor(np.array(first_img))
    
    split_path_str = str(split_path).replace("\\", "/")

    eda_stats = {
        "dataset_name": "Fashion-MNIST",
        "source": "torchvision.datasets.FashionMNIST",
        "license": "MIT License",
        "split": {
            "seed": manifest.split_seed,
            "file": split_path_str,
            "train_size": len(manifest.train_indices),
            "validation_size": len(manifest.validation_indices),
            "test_size": len(manifest.test_indices),
        },
        "image_shape": list(first_tensor.shape),
        "channels": 1,
        "raw_pixel_min": int(first_tensor.min()),
        "raw_pixel_max": int(first_tensor.max()),
        "sample_counts": {
            "total": len(manifest.train_indices) + len(manifest.validation_indices) + len(manifest.test_indices),
            "train": len(manifest.train_indices),
            "validation": len(manifest.validation_indices),
            "test": len(manifest.test_indices),
        },
        "class_counts": {
            "train": {classes[i]: train_counts[i] for i in range(len(classes))},
            "validation": {classes[i]: val_counts[i] for i in range(len(classes))},
            "test": {classes[i]: test_counts[i] for i in range(len(classes))},
        },
    }

    # Save stats JSON
    with open(output_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, indent=2)

    # 3. Class Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, [train_counts[i] for i in range(10)], width, label="Train", color="#1f77b4")
    ax.bar(x, [val_counts[i] for i in range(10)], width, label="Validation", color="#ff7f0e")
    ax.bar(x + width, [test_counts[i] for i in range(10)], width, label="Test", color="#2ca02c")

    ax.set_ylabel("Sample Count")
    ax.set_title("Fashion-MNIST Class Distribution across Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    fig.savefig(output_dir / "class_distribution.png", dpi=200)
    if curated_dir is not None:
        fig.savefig(curated_dir / "class_distribution.png", dpi=200)
    plt.close(fig)

    # 4. Representative samples: one 5x5 grid file per class.
    for class_index, class_name in enumerate(classes):
        class_indices = [
            idx for idx in manifest.train_indices
            if raw_train.targets[idx] == class_index
        ][:25]
        if len(class_indices) < 25:
            raise ValueError(
                f"Class {class_index} ({class_name}) has fewer than 25 training samples."
            )

        fig, axes = plt.subplots(5, 5, figsize=(6, 6))
        for axis, sample_idx in zip(axes.flat, class_indices):
            img, _ = raw_train[sample_idx]
            axis.imshow(img, cmap="gray")
            axis.axis("off")

        fig.suptitle(f"Fashion-MNIST Representative Samples: {class_name}", fontsize=14)
        fig.tight_layout()
        output_path = output_dir / f"representative_class_{class_index}.png"
        fig.savefig(output_path, dpi=200)
        if curated_dir is not None:
            fig.savefig(curated_dir / output_path.name, dpi=200)
        plt.close(fig)
