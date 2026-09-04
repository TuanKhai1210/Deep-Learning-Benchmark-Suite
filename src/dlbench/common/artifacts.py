"""Owner C (Khải): run directories and provenance; never overwrite an existing run."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping


def create_run_dir(output_root: Path, run_id: str) -> Path:
    """Create a new run directory with exist_ok=False; reject unsafe run IDs."""
    raise NotImplementedError("TODO C (Khải): create one unique run directory without overwriting.")


def save_run_metadata(run_dir: Path, config: Mapping[str, Any]) -> None:
    """Save resolved config.json, environment.json, commit/dirty state and hashes.

    Preserve both original Python CONFIG files from config['_sources']; read them
    at run creation. Include split hash and normalization.
    If the worktree is dirty, record a patch/hash or explicitly mark non-release.
    """
    raise NotImplementedError("TODO C (Khải): persist resolved config and reproducibility metadata.")


def append_history(run_dir: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch; stable CSV schema, explicit units, no fabricated values.

    Columns: epoch, train_loss, val_loss, train_accuracy, val_accuracy,
    train_macro_f1, val_macro_f1, learning_rate, epoch_seconds.
    """
    raise NotImplementedError("TODO C (Khải): write/validate one CSV history row.")


def save_metrics(run_dir: Path, metrics: Mapping[str, Any]) -> None:
    """Save metrics.json: evaluation split, scores, epoch, timing scope/units."""
    raise NotImplementedError("TODO C (Khải): serialize real metrics with provenance.")
