"""Owner A (Thiên), reviewer B (Khoa): torchvision source and stable sample IDs.

No download on import. Train/validation use separate wrappers/transform objects,
even when they share the same official training source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST

from dlbench.a1.data.split import create_split, load_split, save_split, validate_split_against_config
from dlbench.a1.data.transforms import compute_normalization

class FashionMNISTSubset(Dataset):
    """Custom wrapper exposing stable sample IDs and applying dedicated transforms."""

    def __init__(
        self,
        base_dataset: FashionMNIST,
        indices: list[int],
        split_name: str,
        transform: Callable | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.split_name = split_name
        self.transform = transform
        self.source_namespace = "official_train" if base_dataset.train else "official_test"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx = self.indices[idx]
        image, label = self.base_dataset[source_idx]

        if self.transform is not None:
            image = self.transform(image)

        sample_id = f"fashion_mnist:{self.source_namespace}:{source_idx}"
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
        }
    
def load_official_dataset(root: str, *, train: bool, download: bool = False) -> FashionMNIST:
    """Return the untransformed FashionMNIST official train or test dataset."""
    return FashionMNIST(root=root, train=train, transform=None, download=download)


def prepare_data(config: MutableMapping[str, Any]) -> dict[str, Any]:
    """Download/check data, create or verify fixed split, measure train-only stats.

    Call split.py and transforms.py. Return measured metadata for group review.
    Do not silently overwrite an existing split or mark a protocol frozen.
    Store provenance and train-only normalization statistics with the split.
    """
    config = config if isinstance(config, dict) else dict(config)
    data_config = config["data"]
    training_config = config["training"]
    run_config = config["run"]

    data_root = data_config["root"]
    split_file = data_config["split_file"]
    split_path = Path(split_file)
    split_seed = data_config["split_seed"]
    val_size = data_config["validation_size"]
    download = data_config["download"]

    # 1. Load official training set without transforms
    raw_train: FashionMNIST = load_official_dataset(data_root, train=True, download=download)
    _ = load_official_dataset(data_root, train=False, download=download)

    # 2. Load or create split manifest
    if split_path.is_file():
        manifest = load_split(split_path)
    else:
        labels = [int(label) for label in raw_train.targets]
        manifest = create_split(labels, validation_size=val_size, split_seed=split_seed)
        save_split(manifest, split_path)
    validate_split_against_config(manifest, data_config)

    # 3. Compute train-only normalization statistics dynamically without leakage
    mean, std = compute_normalization(raw_train, manifest.train_indices)

    preprocessing = config.setdefault("preprocessing", {})
    preprocessing["mean"] = mean
    preprocessing["std"] = std

    metadata = {
        "dataset": "FashionMNIST",
        "split_seed": manifest.split_seed,
        "split_file": str(split_path).replace("\\", "/"),
        "num_train": len(manifest.train_indices),
        "num_val": len(manifest.validation_indices),
        "num_test": len(manifest.test_indices),
        "measured_mean": mean,
        "measured_std": std,
        "classes": raw_train.classes,
    }

    return metadata
