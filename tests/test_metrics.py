"""Unit tests cho module dlbench.a1.metrics."""

import unittest
import torch
import torch.nn as nn

from dlbench.a1.metrics import classification_metrics, count_parameters


class DummyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Input 10 -> Hidden 5 (weights: 50, bias: 5)
        self.fc1 = nn.Linear(10, 5)
        # Hidden 5 -> Output 2 (weights: 10, bias: 2)
        self.fc2 = nn.Linear(5, 2)
        # Đóng băng fc2 để test trainable_parameters
        for param in self.fc2.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class TestMetrics(unittest.TestCase):
    def test_classification_metrics_perfect_score(self) -> None:
        """Đảm bảo accuracy và macro-F1 bằng 1.0 khi nhãn dự đoán trùng khớp hoàn toàn."""
        targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        preds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        res = classification_metrics(targets, preds, num_classes=10)
        self.assertAlmostEqual(res["accuracy"], 1.0)
        self.assertAlmostEqual(res["macro_f1"], 1.0)

    def test_classification_metrics_hand_calculated(self) -> None:
        """Kiểm tra độ chính xác với dữ liệu nhỏ tính tay."""
        # 4 mẫu: target = [0, 0, 1, 1], pred = [0, 1, 1, 1]
        # Lớp 0: TP=1, FP=0, FN=1 -> Precision=1.0, Recall=0.5 -> F1 = 2/3 ≈ 0.6667
        # Lớp 1: TP=2, FP=1, FN=0 -> Precision=2/3, Recall=1.0 -> F1 = 0.8
        # Macro-F1 = (2/3 + 4/5) / 2 = (10/15 + 12/15) / 2 = 11/15 ≈ 0.7333
        # Accuracy = 3 / 4 = 0.75
        targets = [0, 0, 1, 1]
        preds = [0, 1, 1, 1]

        res = classification_metrics(targets, preds, num_classes=2)
        self.assertAlmostEqual(res["accuracy"], 0.75)
        self.assertAlmostEqual(res["macro_f1"], 11 / 15, places=4)

    def test_classification_metrics_zero_division(self) -> None:
        """Kiểm tra xử lý zero_division khi một lớp hoàn toàn không có dự đoán."""
        targets = [0, 0, 1, 1]
        preds = [0, 0, 0, 0]  # Lớp 1 không có TP/FP
        # Lớp 0: Precision=2/4=0.5, Recall=2/2=1.0 -> F1 = 2/3
        # Lớp 1: Precision=0, Recall=0 -> F1 = 0.0 (zero_division=0)
        # Macro-F1 = (2/3 + 0) / 2 = 1/3 ≈ 0.3333
        res = classification_metrics(targets, preds, num_classes=2)
        self.assertAlmostEqual(res["accuracy"], 0.5)
        self.assertAlmostEqual(res["macro_f1"], 1 / 3, places=4)

    def test_classification_metrics_invalid_inputs(self) -> None:
        """Từ chối danh sách rỗng hoặc lệch kích thước."""
        with self.assertRaises(ValueError):
            classification_metrics([], [])

        with self.assertRaises(ValueError):
            classification_metrics([1, 2], [1])

    def test_count_parameters(self) -> None:
        """Kiểm tra đếm đúng total_parameters và trainable_parameters."""
        model = DummyLinearModel()
        # fc1: 10 * 5 + 5 = 55 (trainable)
        # fc2: 5 * 2 + 2 = 12 (frozen)
        # Total: 67, Trainable: 55
        counts = count_parameters(model)
        self.assertEqual(counts["total_parameters"], 67)
        self.assertEqual(counts["trainable_parameters"], 55)


if __name__ == "__main__":
    unittest.main()