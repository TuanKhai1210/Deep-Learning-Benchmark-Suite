"""Owner B (Khoa), reviewer C (Khải): validation-only selection, safe persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    """Compare val_macro_f1 max, val_loss min, epoch min; reject nonfinite metrics."""
    raise NotImplementedError("TODO B (Khoa): deterministic validation-only checkpoint comparison.")


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomic save; weights, epoch, config, split/stats hashes, seed, val metrics.

    Separate best.pt from last.pt. If resume is supported, last.pt also needs
    optimizer/scheduler/RNG state and documented resume granularity.
    """
    raise NotImplementedError("TODO B (Khoa): save complete checkpoint metadata safely.")


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load trusted checkpoint only, then verify expected keys/schema/version."""
    raise NotImplementedError("TODO B (Khoa): load and validate a trusted checkpoint.")
