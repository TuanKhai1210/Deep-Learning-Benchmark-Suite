"""Whole-partition confusion counts; undefined class denominators become null.

Macro scores exclude undefined classes and explicitly report the class count.
Background stays in the matrix, so foreground false positives are counted.
"""
import torch
from .data import CLASS_NAMES


class SegmentationMetrics:
    def __init__(self):
        self.matrix = torch.zeros(17, 17, dtype=torch.int64)
        self.ignored = 0
        self.total = 0

    def update(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")
        target = target.detach().cpu().long().flatten()
        prediction = prediction.detach().cpu().long().flatten()
        if not ((target == 255) | ((target >= 0) & (target < 17))).all():
            raise ValueError("Invalid target labels")
        if not ((prediction >= 0) & (prediction < 17)).all():
            raise ValueError("Invalid prediction labels")
        valid = target != 255
        self.total += target.numel()
        self.ignored += int((~valid).sum())
        self.matrix += torch.bincount(17 * target[valid] + prediction[valid],
                                      minlength=289).reshape(17, 17)

    def compute(self):
        matrix = self.matrix.double()
        tp = matrix.diag()
        support, predicted = matrix.sum(1), matrix.sum(0)
        union = support + predicted - tp
        dice_den = support + predicted
        iou = [float(tp[i] / union[i]) if union[i] else None for i in range(17)]
        dice = [float(2 * tp[i] / dice_den[i]) if dice_den[i] else None for i in range(17)]

        def macro(values, ids):
            present = [values[i] for i in ids if values[i] is not None]
            return sum(present) / len(present) if present else None

        return {
            "foreground_miou": macro(iou, range(1, 17)),
            "foreground_dice": macro(dice, range(1, 17)),
            "trash_miou": macro(iou, range(9, 17)),
            "all_miou": macro(iou, range(17)),
            "all_dice": macro(dice, range(17)),
            "foreground_defined_classes": sum(v is not None for v in iou[1:]),
            "trash_defined_classes": sum(v is not None for v in iou[9:]),
            "ignore_fraction": self.ignored / self.total if self.total else None,
            "valid_pixels": int(matrix.sum()),
            "confusion_matrix": self.matrix.tolist(),
            "per_class": [{"id": i, "name": CLASS_NAMES[i], "iou": iou[i],
                           "dice": dice[i], "support": int(support[i]),
                           "predicted": int(predicted[i])} for i in range(17)],
        }
