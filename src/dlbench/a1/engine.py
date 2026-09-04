"""Owner B (Khoa), reviewer C (Khải): one-epoch mechanics only, shared by every model.

trainer.py owns multi-epoch orchestration, validation checkpoint decisions and
artifacts. Do not add a second training loop inside any model module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dlbench.a1.contracts import EpochMetrics, EvaluationResult

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer,
                    criterion: nn.Module, device: str) -> EpochMetrics:
    """Train mode, move batch, zero_grad, logits, CE, backward, optimizer step.

    Aggregate loss by sample count; compute metrics across epoch predictions.
    Do not apply softmax before CrossEntropyLoss; do not read the test loader.
    """
    raise NotImplementedError("TODO B (Khoa): shared training step and epoch metrics.")


def evaluate_epoch(model: nn.Module, loader: DataLoader,
                   criterion: nn.Module, device: str) -> EvaluationResult:
    """eval + inference_mode; no optimizer update; retain IDs and predictions.

    Aggregate loss by sample count, labels fixed 0..9, macro-F1 over the split.
    Compute probabilities only for outputs/analysis, not as input to CE.
    """
    raise NotImplementedError("TODO B (Khoa): shared evaluation with complete split predictions.")
