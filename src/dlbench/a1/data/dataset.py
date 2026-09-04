"""Owner A (Thiên), reviewer B (Khoa): torchvision source and stable sample IDs.

No download on import. Train/validation use separate wrappers/transform objects,
even when they share the same official training source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset


def load_official_dataset(root: str, *, train: bool, download: bool = False) -> Dataset:
    """Return the untransformed FashionMNIST official train or test dataset."""
    raise NotImplementedError("TODO A (Thiên): load torchvision FashionMNIST only when explicitly called.")


def prepare_data(config: Mapping[str, Any]) -> dict[str, Any]:
    """Download/check data, create or verify fixed split, measure train-only stats.

    Call split.py and transforms.py. Return measured metadata for group review.
    Do not silently overwrite an existing split or mark a protocol frozen.
    Store provenance and train-only normalization statistics with the split.
    """
    raise NotImplementedError("TODO A (Thiên): implement data preparation and return measured metadata.")
