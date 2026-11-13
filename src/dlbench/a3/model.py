"""Trainable heads over frozen image and title features."""

import torch
from torch import nn

MODALITIES = {
    "image-only": ("image",),
    "title-only": ("title",),
    "concat": ("image", "title"),
    "product": ("image", "title"),
    "mean-ablation": ("image", "title"),
}


class ProductClassifier(nn.Module):
    def __init__(self, variant, dimensions, num_classes, projection=128):
        super().__init__()
        self.variant = variant
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dimensions[name], projection), nn.ReLU())
            for name in MODALITIES[variant]
        })
        width = len(self.projections) + int(variant in {"product", "mean-ablation"})
        self.head = nn.Sequential(
            nn.Linear(width * projection, projection), nn.ReLU(),
            nn.Linear(projection, num_classes),
        )

    def forward(self, features):
        projected = [layer(features[name]) for name, layer in self.projections.items()]
        if self.variant == "product":
            projected.append(projected[0] * projected[1])
        elif self.variant == "mean-ablation":
            projected.append((projected[0] + projected[1]) / 2)
        return self.head(torch.cat(projected, dim=1))
