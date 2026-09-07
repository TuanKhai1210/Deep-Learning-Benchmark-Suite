"""Owner A (Thiên), reviewer B (Khoa): canonical preprocessing and train-only augmentation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torchvision import transforms

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def compute_normalization(dataset: Dataset, train_indices: Sequence[int]) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std over pixels in unaugmented TRAIN only.

    Scale uint8 to [0,1] before aggregation. Do not use val/test or random crop.
    Return one mean and one positive std for the grayscale A1 input.
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    num_pixels = 0

    to_tensor = transforms.ToTensor()
    for idx in train_indices:
        # Fetch raw PIL Image or array from dataset without any augmentations
        item = dataset[idx]
        img = item[0] if isinstance(item, (tuple, list)) else item

        # ToTensor scales uint8 [0, 255] to float32 [0.0, 1.0] with shape [1, 28, 28]
        tensor = to_tensor(img) if not isinstance(img, torch.Tensor) else img.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        total_sum += tensor.sum().item()
        total_sq_sum += (tensor ** 2).sum().item()
        num_pixels += tensor.numel()

    mean = total_sum / num_pixels
    variance = (total_sq_sum / num_pixels) - (mean ** 2)
    std = variance ** 0.5

    if std <= 0:
        raise ValueError(f"Computed standard deviation must be positive, got {std}")

    return [float(mean)], [float(std)]

def build_transforms(preprocessing: Mapping[str, Any], *, training: bool) -> Callable:
    """Produce [1,28,28] float32; validation/test transforms must be deterministic.

    Proposed order: optional training-only crop -> ToTensor -> Normalize.
    Never mutate a dataset's transform shared by train and validation Subsets.
    """
    if "mean" not in preprocessing or "std" not in preprocessing:
        raise KeyError(
            "Preprocessing configuration must explicitly provide measured 'mean' and 'std' "
            "from the training split."
        )

    mean = preprocessing["mean"]
    std = preprocessing["std"]

    transform_list: list[Any] = []

    if training:
        if preprocessing.get("random_horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if preprocessing.get("random_crop", False):
            padding = preprocessing.get("crop_padding", 2)
            transform_list.append(transforms.RandomCrop(28, padding=padding))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)
