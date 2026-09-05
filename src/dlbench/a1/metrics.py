"""Owner B (Khoa), reviewer C (Khải): single source of metric definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from sklearn.metrics import accuracy_score, f1_score
if TYPE_CHECKING:
    from torch import nn


def classification_metrics(targets: Sequence[int], predicted_labels: Sequence[int]) -> dict[str, float]:
    """Accuracy and macro_f1 in [0,1]; labels=range(10), zero_division=0.

    Reject length mismatches/empty input. Use the full split, not mean batch F1.
    """
    
    if not targets or not predicted_labels:
        raise ValueError("Input lists cannot be empty!")
    if len(targets) != len(predicted_labels):
        raise ValueError("Lengths of targets and predicted labels must match")
    
    accuracy = accuracy_score(targets, predicted_labels)
    macro_f1 = f1_score(targets, predicted_labels, average='macro')
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1)
    }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total_parameters and trainable_parameters, not an estimate."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable
    }
