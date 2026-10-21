"""Owner A (Thiên): CNNClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn


class CNNClassifier(nn.Module):
    """Classify 28x28 grayscale images with a small convolutional network."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()

        num_classes = parameters.get("num_classes")
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")

        channels = parameters.get("channels")
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError("channels must be a non-empty list of positive integers")
        if any(type(channel) is not int or channel <= 0 for channel in channels):
            raise ValueError("channels must be a non-empty list of positive integers")

        hidden_dim = parameters.get("hidden_dim", 128)
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")

        dropout = parameters.get("dropout", 0.0)
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a number in the range [0, 1)")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be a number in the range [0, 1)")

        feature_layers: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            feature_layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                ]
            )
            input_channels = output_channels

        self.features = nn.Sequential(*feature_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        Keep spatial image input and return unnormalized class scores.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("images must have shape [batch, 1, 28, 28]")
        if images.size(0) == 0:
            raise ValueError("batch size cannot be zero")

        features = self.pool(self.features(images))
        return self.classifier(features)
