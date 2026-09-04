"""Owner B (Khoa): RNNClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from torch import Tensor, nn


class RNNClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. For rows: [B,1,28,28] -> [B,28,28], batch_first=True. Document timestep and hidden-state selection."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        # TODO B (Khoa): validate config and declare the agreed LSTM OR GRU, plus classification head.
        # No optimizer, DataLoader, training loop or hard-coded device here.
        raise NotImplementedError("TODO B (Khoa): implement RNNClassifier layers.")

    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        For rows: [B,1,28,28] -> [B,28,28], batch_first=True. Document timestep and hidden-state selection.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        raise NotImplementedError("TODO B (Khoa): implement RNNClassifier.forward.")
