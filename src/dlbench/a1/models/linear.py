"""Owner A (Thiên): LinearClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from torch import Tensor, nn


class LinearClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. Flatten internally; return unnormalized logits. No hidden layer or softmax."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        if "input_dim" not in parameters:
            raise ValueError("Input dimension not specified")
        if "num_classes" not in parameters:
            raise ValueError("Number of output classes not specified")
        
        input_dim = parameters["input_dim"]
        num_classes = parameters["num_classes"]

        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("Input dimension must be a positive integer")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("Number of output classes must be a positive integer")

        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        Flatten internally; return unnormalized logits. No hidden layer or softmax.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        if images.ndim < 2:
            raise ValueError("Input must include a batch dimension")
        if images.size(0) == 0:
            raise ValueError("Batch size cannot be zero")
        return self.network(images)
