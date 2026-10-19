"""Owner C (Khải): Transformer classifier with the shared image/logit contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention over image-patch tokens."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, num_patches, _ = tokens.shape

        q = self.q_proj(tokens).reshape(
            batch, num_patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(tokens).reshape(
            batch, num_patches, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(tokens).reshape(
            batch, num_patches, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        weights = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
        context = weights @ v
        combined = context.transpose(1, 2).reshape(
            batch, num_patches, self.embed_dim
        )
        return self.out_proj(combined)


class TransformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.attention_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = 4 * embed_dim
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention_dropout(self.attention(self.norm1(tokens)))
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class TransformerClassifier(nn.Module):
    """Classify 28x28 grayscale images using patch tokens and self-attention.

    Non-overlapping patches are projected to embed_dim, combined with learned
    position embeddings, then processed by Transformer blocks. Mean pooling
    over patch tokens produces one vector for the classification head.
    """

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        patch_size = parameters["patch_size"]
        if type(patch_size) is not int or patch_size <= 0 or 28 % patch_size != 0:
            raise ValueError("patch_size must be a positive integer dividing 28")
        self.patch_size = patch_size
        self.embed_dim = parameters["embed_dim"]
        num_heads = parameters["num_heads"]
        num_layers = parameters["num_layers"]
        num_classes = parameters["num_classes"]
        dropout = parameters["dropout"]

        for name, value in (
            ("embed_dim", self.embed_dim),
            ("num_heads", num_heads),
            ("num_layers", num_layers),
            ("num_classes", num_classes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if type(dropout) not in (int, float) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        num_patches = (28 // self.patch_size) ** 2

        self.patch_projection = nn.Linear(
            in_features=self.patch_size ** 2,
            out_features=self.embed_dim,
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, self.embed_dim) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def patchify(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(
                f"expected images [B, 1, 28, 28], got {tuple(images.shape)}"
            )
        if images.dtype != torch.float32:
            raise TypeError(f"expected float32 images, got {images.dtype}")

        batch, _, height, width = images.shape
        p = self.patch_size

        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.reshape(batch, (height // p) * (width // p), p * p)
        return patches

    def embed_patches(self, images: Tensor) -> Tensor:
        patches = self.patchify(images)
        tokens = self.patch_projection(patches)
        return tokens + self.position_embedding

    def forward(self, images: Tensor) -> Tensor:
        """Return raw [B, num_classes] logits for float32 [B, 1, 28, 28] images."""
        tokens = self.embed_patches(images)
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.final_norm(tokens)
        image_vector = tokens.mean(dim=1)
        return self.classifier(image_vector)
