"""Cross entropy plus foreground soft Dice, with ignore removed from both.

Dice averages target-present foreground classes over each batch. Empty foreground
batches have zero Dice loss; valid background still contributes cross entropy.
Epoch loss is a valid-pixel-weighted mean of these batch objectives.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight, self.dice_weight = ce_weight, dice_weight

    def forward(self, logits, target):
        if logits.shape[:2] != (target.shape[0], 17) or logits.shape[2:] != target.shape[1:]:
            raise ValueError("Expected logits [B,17,H,W] and mask [B,H,W]")
        valid = target != 255
        if not valid.any():
            return logits.sum() * 0
        ce = F.cross_entropy(logits, target, ignore_index=255)
        safe = target.masked_fill(~valid, 0)
        truth = F.one_hot(safe, 17).permute(0, 3, 1, 2).to(logits.dtype)
        truth = truth * valid[:, None]
        probs = logits.softmax(1) * valid[:, None]
        axes = (0, 2, 3)
        support = truth.sum(axes)[1:]
        intersection = (truth * probs).sum(axes)[1:]
        denominator = (truth + probs).sum(axes)[1:]
        present = support > 0
        dice = (1 - (2 * intersection[present] + 1e-6) /
                (denominator[present] + 1e-6)).mean() if present.any() else logits.sum() * 0
        return self.ce_weight * ce + self.dice_weight * dice
