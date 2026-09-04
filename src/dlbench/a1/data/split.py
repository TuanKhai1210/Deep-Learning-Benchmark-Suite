"""Owner A (Thiên), reviewer B (Khoa): one persisted partition shared by every model."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from dlbench.a1.contracts import SplitManifest


def create_split(labels: Sequence[int], *, validation_size: int, split_seed: int) -> SplitManifest:
    """Stratify the OFFICIAL training set; never mix official test into train/val."""
    raise NotImplementedError("TODO A (Thiên): stratified split, official test indices, exact counts.")


def validate_split(manifest: SplitManifest) -> None:
    """Check ranges, duplicates, 60k partition coverage, and 10k test coverage.

    Train/val IDs share an index namespace. Official test IDs do NOT; a test
    index 123 is not the same source sample as official training index 123.
    """
    raise NotImplementedError("TODO A (Thiên): verify split integrity and source namespaces.")


def save_split(manifest: SplitManifest, path: Path) -> None:
    """Write JSON plus schema/version/provenance; refuse silent replacement."""
    raise NotImplementedError("TODO A (Thiên): persist a reviewed split and compute its hash.")


def load_split(path: Path) -> SplitManifest:
    """Read the existing manifest and validate it; never resplit during training."""
    raise NotImplementedError("TODO A (Thiên): deserialize and validate the fixed split.")
