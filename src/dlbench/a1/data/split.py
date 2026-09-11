"""Owner A (Thiên), reviewer B (Khoa): one persisted partition shared by every model."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from dlbench.a1.contracts import SplitManifest


def create_split(labels: Sequence[int], *, validation_size: int, split_seed: int) -> SplitManifest:
    """Stratify the OFFICIAL training set; never mix official test into train/val."""
    labels_arr = np.asarray(labels)
    num_samples = len(labels_arr)

    if validation_size <= 0 or validation_size >= num_samples:
        raise ValueError(f"validation_size must be between 1 and {num_samples - 1}, got {validation_size}")

    rng = np.random.default_rng(split_seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    classes = np.unique(labels_arr)
    for c in classes:
        cls_idx = np.where(labels_arr == c)[0]
        rng.shuffle(cls_idx)
        cls_val_count = int(round(len(cls_idx) * (validation_size / num_samples))) # Proportional split for each class
        val_indices.extend(cls_idx[:cls_val_count].tolist())
        train_indices.extend(cls_idx[cls_val_count:].tolist())

    train_indices.sort()
    val_indices.sort()
    test_indices = list(range(10_000))

    manifest = SplitManifest(
        dataset="fashion_mnist",
        split_seed=split_seed,
        train_indices=train_indices,
        validation_indices=val_indices,
        test_indices=test_indices,
    )

    validate_split(manifest)
    return manifest


def validate_split(manifest: SplitManifest) -> None:
    """Check ranges, duplicates, 60k partition coverage, and 10k test coverage."""
    train_set = set(manifest.train_indices)
    val_set = set(manifest.validation_indices)
    test_set = set(manifest.test_indices)

    # 1. No duplicates within splits
    if len(train_set) != len(manifest.train_indices):
        raise ValueError("Duplicate indices found in train_indices")
    if len(val_set) != len(manifest.validation_indices):
        raise ValueError("Duplicate indices found in validation_indices")
    if len(test_set) != len(manifest.test_indices):
        raise ValueError("Duplicate indices found in test_indices")

    # 2. Train and Val must be disjoint
    overlap = train_set.intersection(val_set)
    if overlap:
        raise ValueError(f"Data leakage: train and validation share {len(overlap)} indices: {list(overlap)[:5]}")

    # 3. Exactly partition the 60,000 official training images
    combined_train_val = train_set.union(val_set)
    if combined_train_val != set(range(60_000)):
        raise ValueError(
            f"Train and validation must partition indices 0..59999. Found {len(combined_train_val)} total samples."
        )

    # 4. Cover the 10,000 official test set images
    if test_set != set(range(10_000)):
        raise ValueError(f"Test indices must span indices 0..9999. Found {len(test_set)} elements.")


def save_split(manifest: SplitManifest, path: Path) -> None:
    """Write JSON plus schema/version/provenance; refuse silent replacement."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Split file already exists at '{path}'.")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": manifest.dataset,
        "split_seed": manifest.split_seed,
        "version": "1.0",
        "counts": {
            "train": len(manifest.train_indices),
            "validation": len(manifest.validation_indices),
            "test": len(manifest.test_indices),
        },
        "train_indices": manifest.train_indices,
        "validation_indices": manifest.validation_indices,
        "test_indices": manifest.test_indices,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split(path: Path) -> SplitManifest:
    """Read the existing manifest and validate it; never resplit during training."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = SplitManifest(
        dataset=data.get("dataset", "fashion_mnist"),
        split_seed=data["split_seed"],
        train_indices=data["train_indices"],
        validation_indices=data["validation_indices"],
        test_indices=data["test_indices"],
    )
    validate_split(manifest)
    return manifest
