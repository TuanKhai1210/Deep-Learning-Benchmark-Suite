"""Pinned ADE20K B0 initialization with a newly initialized 17-class head.

Preprocessing is external: ImageNet normalization, no reduce_labels and no second
resize. Interpolate raw logits to the common target canvas before loss/metrics.
"""
from torch import nn
from torch.nn import functional as F
from ..data import CLASS_NAMES


class SegFormer(nn.Module):
    def __init__(self, checkpoint, revision, *, pretrained=True):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise ImportError("SegFormer requires the 'a2' optional dependencies: pip install -e .[a2]") from error
        labels = {i: name for i, name in enumerate(CLASS_NAMES)}
        if pretrained:
            self.network = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint, revision=revision, num_labels=17,
                id2label=labels, label2id={name: i for i, name in labels.items()},
                ignore_mismatched_sizes=True,
            )
        else:
            # Explicit development path. Never silently fall back after download failure.
            self.network = SegformerForSemanticSegmentation(SegformerConfig(
                num_labels=17, id2label=labels, label2id={n: i for i, n in labels.items()},
            ))

    def forward(self, image):
        logits = self.network(pixel_values=image).logits
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
