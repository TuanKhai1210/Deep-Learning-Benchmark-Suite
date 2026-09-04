"""Owner C (Khải): TransformerClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from torch import Tensor, nn


class TransformerClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. For patch_size=4: 49 patches, each 16 pixels before projection. Specify pooling/CLS policy; embed_dim divisible by num_heads."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        # TODO C (Khải): validate config and declare patch projection, positional encoding, encoder and classifier.
        # No optimizer, DataLoader, training loop or hard-coded device here.
        raise NotImplementedError("TODO C (Khải): implement TransformerClassifier layers.")

    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        For patch_size=4: 49 patches, each 16 pixels before projection. Specify pooling/CLS policy; embed_dim divisible by num_heads.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        raise NotImplementedError("TODO C (Khải): implement TransformerClassifier.forward.")
