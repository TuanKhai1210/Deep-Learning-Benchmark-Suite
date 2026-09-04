"""Owner A (Thiên), reviewer B (Khoa): canonical preprocessing and train-only augmentation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def compute_normalization(dataset: Dataset, train_indices: Sequence[int]) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std over pixels in unaugmented TRAIN only.

    Scale uint8 to [0,1] before aggregation. Do not use val/test or random crop.
    Return one mean and one positive std for the grayscale A1 input.
    """
    raise NotImplementedError("TODO A (Thiên): compute train-only normalization without leakage.")


def build_transforms(preprocessing: Mapping[str, Any], *, training: bool) -> Callable:
    """Produce [1,28,28] float32; validation/test transforms must be deterministic.

    Proposed order: optional training-only crop -> ToTensor -> Normalize.
    Never mutate a dataset's transform shared by train and validation Subsets.
    """
    raise NotImplementedError("TODO A (Thiên): separate train and evaluation transforms.")
