"""Owner B (Khoa): MLPClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from torch import Tensor, nn


class MLPClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. Flatten internally; explain activation and regularization in the method note."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        # TODO B (Khoa): validate config and declare hidden layers from hidden_dims, activation/dropout and output head.
        # No optimizer, DataLoader, training loop or hard-coded device here.
        raise NotImplementedError("TODO B (Khoa): implement MLPClassifier layers.")

    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        Flatten internally; explain activation and regularization in the method note.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        raise NotImplementedError("TODO B (Khoa): implement MLPClassifier.forward.")
