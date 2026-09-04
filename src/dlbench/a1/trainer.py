"""Owner B (Khoa), reviewer C (Khải): assemble data, model, optimizer and the shared engine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from dlbench.a1.contracts import EvaluationResult, FitResult


def fit(config: Mapping[str, Any], *, smoke: bool = False) -> FitResult:
    """Seed -> loaders -> model -> optimizer -> train/val epochs -> artifacts.

    Main runs require strict config validation AND valid existing split/stats.
    Smoke runs: explicitly limited training/validation data, positive small
    epoch/batch budget, separate smoke directory, no official-test evaluation.
    Call engine.py, checkpoint.py and common/artifacts.py; do not duplicate them.
    Save resolved config and actual epoch/time budget. Never overwrite a run.
    """
    raise NotImplementedError("TODO B (Khoa): orchestrate shared training and validation checkpointing.")


def evaluate_checkpoint(config: Mapping[str, Any], checkpoint_path: Path, *,
                        split: Literal["validation", "test"] = "validation") -> EvaluationResult:
    """Reconstruct from saved config; verify split hash, normalization and class order.

    Refuse incompatible supplied config. Evaluate the requested split with
    engine.evaluate_epoch; preserve checkpoint provenance. Test use is explicit
    and only after all model-selection decisions are fixed.
    Load only trusted checkpoints using an appropriate safe loading policy.
    """
    raise NotImplementedError("TODO B (Khoa): load/verify checkpoint and evaluate without tuning.")
