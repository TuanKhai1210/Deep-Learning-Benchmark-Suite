"""Owner A (Thiên), reviewer B (Khoa): Batch and DataLoaders contracts."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dlbench.a1.contracts import Batch, DataLoaders
from dlbench.a1.data.dataset import FashionMNISTSubset, load_official_dataset
from dlbench.a1.data.split import load_split
from dlbench.a1.data.transforms import build_transforms
from dlbench.common.reproducibility import seed_worker

def collate_samples(samples: Sequence[Mapping[str, Any]]) -> Batch:
    """Stack image/label tensors; preserve sample_ids as list[str]."""
    images = torch.stack([sample["image"] for sample in samples], dim=0)
    labels = torch.stack([sample["label"] for sample in samples], dim=0)
    sample_ids = [str(sample["sample_id"]) for sample in samples]

    return Batch(images=images, labels=labels, sample_ids=sample_ids)

def build_dataloaders(config: Mapping[str, Any], *, smoke: bool = False) -> DataLoaders:
    """Read existing split; never make a new random split inside this function.

    Train: shuffle with dedicated run-seeded generator and seed_worker.
    Val/test: no random transforms, no shuffle, drop_last=False.
    A smoke subset must be explicitly tagged as smoke, not a benchmark result.
    A/B/C must agree how its small positive batch size is chosen from draft config.
    """
    # Extract nested data section if present, else fall back to root config
    data_cfg = config.get("data", {})
    data_root = str(data_cfg.get("root", config.get("data_root", "./data")))
    split_path = Path(
        data_cfg.get("split_file", config.get("split_path", "configs/a1/splits/fashion_mnist_seed36.json"))
    )
    manifest = load_split(split_path)

    # Load raw official datasets without transforms attached to base instances
    raw_train = load_official_dataset(data_root, train=True, download=False)
    raw_test = load_official_dataset(data_root, train=False, download=False)

    # Preprocessing and augmentation specs
    preprocessing = config.get("preprocessing", {})
    train_transform = build_transforms(preprocessing, training=True)
    eval_transform = build_transforms(preprocessing, training=False)

    train_indices = manifest.train_indices
    val_indices = manifest.validation_indices
    test_indices = manifest.test_indices

    # Handle smoke mode: select a minimal subset for quick execution testing
    if smoke:
        smoke_size = int(config.get("smoke_samples", 64))
        train_indices = train_indices[:smoke_size]
        val_indices = val_indices[:smoke_size]
        test_indices = test_indices[:smoke_size]

    # Wrap subsets with independent transforms and sample ID annotations
    train_dataset = FashionMNISTSubset(raw_train, train_indices, split_name="train", transform=train_transform)
    val_dataset = FashionMNISTSubset(raw_train, val_indices, split_name="validation", transform=eval_transform)
    test_dataset = FashionMNISTSubset(raw_test, test_indices, split_name="test", transform=eval_transform)

    # Hardware & performance settings
    batch_size = int(config.get("batch_size", 64)) if not smoke else min(16, len(train_indices))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))

    # Dedicated generator for reproducible shuffling in training
    # Default development run seed is 69420 per experiment contract
    run_seed = int(config.get("run_seed", 69420))
    train_generator = torch.Generator()
    train_generator.manual_seed(run_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_samples,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_samples,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_samples,
        drop_last=False,
    )

    return DataLoaders(
        train=train_loader,
        validation=val_loader,
        test=test_loader,
    )
