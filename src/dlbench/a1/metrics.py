"""Owner B (Khoa), reviewer C (Khải): single source of metric definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def classification_metrics(targets: Sequence[int], predicted_labels: Sequence[int]) -> dict[str, float]:
    """Accuracy and macro_f1 in [0,1]; labels=range(10), zero_division=0.

    Reject length mismatches/empty input. Use the full split, not mean batch F1.
    """
    raise NotImplementedError("TODO B (Khoa): accuracy and full-split macro-F1.")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total_parameters and trainable_parameters, not an estimate."""
    raise NotImplementedError("TODO B (Khoa): count parameter.numel(), distinguish requires_grad.")
