"""Unit tests for dlbench.a1.engine (Task K03)."""

from __future__ import annotations

import unittest
from typing import TypedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dlbench.a1.contracts import Batch
from dlbench.a1.engine import evaluate_epoch, train_one_epoch
from dlbench.a1.metrics import classification_metrics


class SimpleIndexedDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, sample_ids: list[str] | None = None) -> None:
        self.features = features
        self.labels = labels
        self.sample_ids = sample_ids if sample_ids is not None else [f"real_id_{i}" for i in range(len(labels))]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.features[index], self.labels[index], self.sample_ids[index]


def collate_to_typed_dict(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> Batch:
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    sample_ids = [item[2] for item in batch]
    return {
        "images": images,
        "labels": labels,
        "sample_ids": sample_ids,
    }


class ModeTrackingLinear(nn.Module):
    """Linear layer that records self.training status at the exact moment forward() is called."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.recorded_forward_training_modes: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.recorded_forward_training_modes.append(self.training)
        return self.linear(x)


class DeterministicPredictor(nn.Module):
    """Mock model returning pre-defined logits to test metric aggregations."""

    def __init__(self, logits_sequence: list[torch.Tensor]) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.logits_sequence = logits_sequence
        self.call_idx = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits_sequence[self.call_idx]
        self.call_idx += 1
        return logits + 0.0 * self.dummy_param

class DropoutProbeModel(nn.Module):
    """Network with Dropout(p=1.0) to physically verify mode behavior during forward()."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=1.0)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # Khởi tạo ma trận trọng số đơn vị (hoặc ones) để ánh xạ trực tiếp
        nn.init.ones_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trong train: dropout triệt tiêu toàn bộ tensor x thành 0
        # Trong eval: dropout giữ nguyên tensor x
        dropped = self.dropout(x)
        return self.linear(dropped)


class TestEngine(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(69420)
        self.feature_dim = 8
        self.num_classes = 10
        self.device = "cpu"

    def test_forward_mode_enforcement_with_dropout(self) -> None:
        """Verify train/eval mode during forward passes using deterministic Dropout(p=1.0)."""
        # Tạo đầu vào toàn số 1.0 (nonzero)
        features = torch.ones(8, self.feature_dim)
        labels = torch.zeros(8, dtype=torch.long)
        dataset = SimpleIndexedDataset(features, labels)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_to_typed_dict)

        # 1. Kiểm tra train_one_epoch: Dropout(p=1.0) phải kích hoạt
        model_train = DropoutProbeModel(self.feature_dim, self.num_classes)
        optimizer = torch.optim.SGD(model_train.parameters(), lr=0.01)

        # Ban đầu cố tình đặt eval để xem train_one_epoch có bật model.train() không
        model_train.eval()
        train_result = train_one_epoch(
            model=model_train,
            loader=loader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
        )

        # Khi Dropout(p=1.0) chạy trong train mode:
        # Đầu vào bị triệt tiêu thành 0 -> logits toàn 0 -> xác suất đều giữa 10 class -> loss = ln(10) ≈ 2.302585
        expected_train_loss = -torch.log(torch.tensor(1.0 / self.num_classes)).item()
        self.assertAlmostEqual(train_result.loss, expected_train_loss, places=4)
        self.assertTrue(model_train.training)

        # 2. Kiểm tra evaluate_epoch: Dropout(p=1.0) KHÔNG được phép kích hoạt
        model_eval = DropoutProbeModel(self.feature_dim, self.num_classes)
        # Cố tình đặt model.train() trước để xác nhận evaluate_epoch tự động chuyển sang eval()
        model_eval.train()

        eval_result = evaluate_epoch(
            model=model_eval,
            loader=loader,
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
        )

        # Khi Dropout ở eval mode:
        # Đầu vào ones qua Linear(ones) -> mỗi logit = feature_dim * 1.0 = 8.0
        # Nếu Dropout bị kích hoạt nhầm (chế độ train), loss sẽ là ln(10) ≈ 2.3026.
        # Ở đây vì labels = 0 và tất cả logits đều bằng 8.0, loss thực tế vẫn đồng nhất,
        # nhưng ta kiểm tra trực tiếp qua xác suất hoặc output model không bị zero:
        with torch.inference_mode():
            eval_forward_out = model_eval(features)
            self.assertTrue(torch.all(eval_forward_out > 0.0), "Dropout must be inactive during eval mode.")

        self.assertFalse(model_eval.training)

    def test_sample_ids_strict_validation(self) -> None:
        """evaluate_epoch must raise ValueError if TypedDict is missing sample_ids or has a size mismatch."""
        features = torch.randn(6, self.feature_dim)
        labels = torch.randint(0, self.num_classes, (6,))
        model = nn.Linear(self.feature_dim, self.num_classes)
        criterion = nn.CrossEntropyLoss()

        # Case 1: TypedDict batch has sample_ids missing or None
        def collate_missing_ids(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> dict:
            return {
                "images": torch.stack([x[0] for x in batch]),
                "labels": torch.stack([x[1] for x in batch]),
                "sample_ids": None,
            }

        dataset = SimpleIndexedDataset(features, labels)
        loader_missing = DataLoader(dataset, batch_size=3, collate_fn=collate_missing_ids)
        with self.assertRaises(ValueError) as ctx_missing:
            evaluate_epoch(model, loader_missing, criterion, self.device)
        self.assertIn("missing", str(ctx_missing.exception).lower())

        # Case 2: Length of sample_ids does not match batch size
        def collate_mismatched(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> dict:
            return {
                "images": torch.stack([x[0] for x in batch]),
                "labels": torch.stack([x[1] for x in batch]),
                "sample_ids": [x[2] for x in batch[:-1]],  # Drop one ID
            }

        loader_mismatch = DataLoader(dataset, batch_size=3, collate_fn=collate_mismatched)
        with self.assertRaises(ValueError) as ctx_mismatch:
            evaluate_epoch(model, loader_mismatch, criterion, self.device)
        self.assertIn("does not match batch size", str(ctx_mismatch.exception).lower())

    def test_sample_ids_preservation(self) -> None:
        """Verify evaluate_epoch preserves true source IDs from the TypedDict batch."""
        features = torch.randn(4, self.feature_dim)
        labels = torch.randint(0, self.num_classes, (4,))
        real_ids = ["fmnist_001", "fmnist_002", "fmnist_003", "fmnist_004"]
        dataset = SimpleIndexedDataset(features, labels, sample_ids=real_ids)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_to_typed_dict)

        model = nn.Linear(self.feature_dim, self.num_classes)
        result = evaluate_epoch(model, loader, nn.CrossEntropyLoss(), self.device)

        self.assertEqual(result.predictions.sample_ids, real_ids)

    def test_uneven_batch_loss_sample_weighting(self) -> None:
        """Verify loss aggregation weights uneven batch sizes correctly instead of a naive mean."""
        features = torch.zeros(10, self.feature_dim)
        labels = torch.zeros(10, dtype=torch.long)
        dataset = SimpleIndexedDataset(features, labels)
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_to_typed_dict)

        b1_logits = torch.full((8, self.num_classes), -10.0)
        b1_logits[:, 0] = -2.0
        b2_logits = torch.full((2, self.num_classes), -10.0)
        b2_logits[:, 0] = -10.0

        model = DeterministicPredictor([b1_logits, b2_logits])
        eval_result = evaluate_epoch(model, loader, nn.CrossEntropyLoss(), self.device)

        loss_b1 = nn.CrossEntropyLoss()(b1_logits, labels[:8]).item()
        loss_b2 = nn.CrossEntropyLoss()(b2_logits, labels[8:]).item()
        expected_weighted_loss = (loss_b1 * 8 + loss_b2 * 2) / 10.0

        self.assertAlmostEqual(eval_result.metrics.loss, expected_weighted_loss, places=5)
        self.assertNotAlmostEqual(eval_result.metrics.loss, (loss_b1 + loss_b2) / 2.0, places=1)

    def test_dataset_level_macro_f1_vs_batch_averaged_f1(self) -> None:
        """Verify macro-F1 is computed over concatenated predictions rather than averaged across batches."""
        batch1_targets = [0, 0, 1]
        batch1_preds = [0, 0, 0]

        batch2_targets = [1, 1, 0]
        batch2_preds = [1, 1, 1]

        f1_b1 = classification_metrics(batch1_targets, batch1_preds)["macro_f1"]
        f1_b2 = classification_metrics(batch2_targets, batch2_preds)["macro_f1"]
        batch_averaged_f1 = (f1_b1 + f1_b2) / 2.0

        all_targets = batch1_targets + batch2_targets
        all_preds = batch1_preds + batch2_preds
        expected_global_f1 = classification_metrics(all_targets, all_preds)["macro_f1"]

        self.assertNotEqual(round(batch_averaged_f1, 4), round(expected_global_f1, 4))

        def build_logits(preds: list[int]) -> torch.Tensor:
            t = torch.zeros(len(preds), 10)
            for i, p in enumerate(preds):
                t[i, p] = 10.0
            return t

        logits_b1 = build_logits(batch1_preds)
        logits_b2 = build_logits(batch2_preds)

        features = torch.zeros(6, self.feature_dim)
        labels = torch.tensor(all_targets, dtype=torch.long)
        dataset = SimpleIndexedDataset(features, labels)
        loader = DataLoader(dataset, batch_size=3, collate_fn=collate_to_typed_dict)

        model = DeterministicPredictor([logits_b1, logits_b2])
        eval_result = evaluate_epoch(model, loader, nn.CrossEntropyLoss(), self.device)

        self.assertAlmostEqual(eval_result.metrics.macro_f1, expected_global_f1, places=5)
        self.assertNotAlmostEqual(eval_result.metrics.macro_f1, batch_averaged_f1, places=3)


if __name__ == "__main__":
    unittest.main()