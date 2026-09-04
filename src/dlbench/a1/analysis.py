"""Owner A (Thiên) for error plots; B for comparisons; C for report integration."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from dlbench.a1.contracts import Predictions


def save_predictions(predictions: Predictions, path: Path) -> None:
    """CSV: sample_id, target, predicted_label, then probability_0..probability_9."""
    raise NotImplementedError("TODO B (Khoa): export predictions with stable original sample IDs.")


def analyze_run(run_dir: Path) -> None:
    """Use saved history/predictions; do not retrain or pick a new checkpoint.

    TODO A (Thiên): fixed-order confusion matrix, learning curves, correct/error samples.
    TODO C (Khải): preserve run provenance and explicit train/val/test labels on plots.
    Only figures selected after review should be copied to docs/assets/a1.
    """
    raise NotImplementedError("TODO A/C (Thiên/Khải): analyze genuine saved artifacts.")


def compare_runs(run_dirs: Sequence[Path], output_path: Path) -> None:
    """TODO B (Khoa): assert matching split/protocol/timing before tabulating metrics.

    Report mean/std over the SAME seeds if repeated; no cherry-picked best seed.
    Disclose hardware differences rather than ranking incompatible timing values.
    """
    raise NotImplementedError("TODO B (Khoa): aggregate compatible runs into a comparison table.")
