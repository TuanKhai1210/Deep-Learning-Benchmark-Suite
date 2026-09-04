"""Owner A (Thiên), reviewer B (Khoa): Batch and DataLoaders contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from dlbench.a1.contracts import Batch, DataLoaders


def collate_samples(samples: Sequence[Mapping[str, Any]]) -> Batch:
    """Stack image/label tensors; preserve sample_ids as list[str]."""
    raise NotImplementedError("TODO A (Thiên): collate exactly images, labels, sample_ids.")


def build_dataloaders(config: Mapping[str, Any], *, smoke: bool = False) -> DataLoaders:
    """Read existing split; never make a new random split inside this function.

    Train: shuffle with dedicated run-seeded generator and seed_worker.
    Val/test: no random transforms, no shuffle, drop_last=False.
    A smoke subset must be explicitly tagged as smoke, not a benchmark result.
    A/B/C must agree how its small positive batch size is chosen from draft config.
    """
    raise NotImplementedError("TODO A (Thiên): build train/validation/test loaders with a fixed split.")
