"""Owner A (Thiên): LinearClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from torch import Tensor, nn


class LinearClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. Flatten internally; return unnormalized logits. No hidden layer or softmax."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        # TODO A (Thiên): validate config and declare a linear 784 -> num_classes layer.
        # No optimizer, DataLoader, training loop or hard-coded device here.
        raise NotImplementedError("TODO A (Thiên): implement LinearClassifier layers.")

    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        Flatten internally; return unnormalized logits. No hidden layer or softmax.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        raise NotImplementedError("TODO A (Thiên): implement LinearClassifier.forward.")
