"""Owner A (Thiên): CNNClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from torch import Tensor, nn


class CNNClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. Keep spatial image input; derive classifier dimensions, do not guess or use pretrained-only."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        # TODO A (Thiên): validate config and declare your own conv/pool feature extractor and classifier head.
        # No optimizer, DataLoader, training loop or hard-coded device here.
        raise NotImplementedError("TODO A (Thiên): implement CNNClassifier layers.")

    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        Keep spatial image input; derive classifier dimensions, do not guess or use pretrained-only.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        raise NotImplementedError("TODO A (Thiên): implement CNNClassifier.forward.")
