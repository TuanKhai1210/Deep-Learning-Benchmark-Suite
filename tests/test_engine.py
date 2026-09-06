"""Unit tests for dlbench.a1.engine (Task K03)."""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dlbench.a1.contracts import Batch, EpochMetrics, EvaluationResult
from dlbench.a1.engine import evaluate_epoch, train_one_epoch


class MockBatchDataset(Dataset):
    """Dataset whose samples provide the fields required by Batch."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.images[index], self.labels[index], f"sample_{index}"


def batch_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> Batch:
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    sample_ids = [item[2] for item in batch]
    return {"images": images, "labels": labels, "sample_ids": sample_ids}


class TestEngine(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(69420)
        # Synthetic feature dimension = 8, 10 output classes
        self.num_samples = 32
        self.feature_dim = 8
        self.num_classes = 10
        self.device = "cpu"

        self.features = torch.randn(self.num_samples, self.feature_dim)
        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))

        self.dataset = MockBatchDataset(self.features, self.labels)
        self.loader = DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=batch_collate,
        )

        self.model = nn.Linear(self.feature_dim, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_train_one_epoch_updates_weights_and_returns_epoch_metrics(self) -> None:
        """Verify train_one_epoch executes forward/backward steps and modifies model parameters."""
        initial_weight = self.model.weight.clone()

        result = train_one_epoch(
            model=self.model,
            loader=self.loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
        )

        # Assert returned contract
        self.assertIsInstance(result, EpochMetrics)
        self.assertEqual(result.num_samples, self.num_samples)
        self.assertGreater(result.loss, 0.0)
        self.assertTrue(0.0 <= result.accuracy <= 1.0)
        self.assertTrue(0.0 <= result.macro_f1 <= 1.0)

        # Assert model weights were altered by gradient update
        self.assertFalse(
            torch.equal(initial_weight, self.model.weight),
            "Weights should be updated after train_one_epoch step.",
        )
        self.assertTrue(self.model.training, "Model should remain in training mode.")

    def test_evaluate_epoch_does_not_alter_weights_and_leaves_model_in_eval(self) -> None:
        """Verify evaluate_epoch preserves model weights without gradient side effects."""
        initial_weight = self.model.weight.clone()

        result = evaluate_epoch(
            model=self.model,
            loader=self.loader,
            criterion=self.criterion,
            device=self.device,
        )

        # Assert contract types
        self.assertIsInstance(result, EvaluationResult)
        self.assertIsInstance(result.metrics, EpochMetrics)

        # Assert weights remain intact
        self.assertTrue(
            torch.equal(initial_weight, self.model.weight),
            "Weights must not change during evaluate_epoch.",
        )
        self.assertFalse(self.model.training, "Model must remain in evaluation mode.")

    def test_evaluate_epoch_predictions_structure(self) -> None:
        """Verify predictions payload contains required sample_ids, targets, and probabilities."""
        result = evaluate_epoch(
            model=self.model,
            loader=self.loader,
            criterion=self.criterion,
            device=self.device,
        )

        preds = result.predictions
        self.assertEqual(len(preds.sample_ids), self.num_samples)
        self.assertEqual(len(preds.targets), self.num_samples)
        self.assertEqual(len(preds.predicted_labels), self.num_samples)
        self.assertEqual(len(preds.probabilities), self.num_samples)

        # Probability row must sum to approximately 1.0 across classes
        first_prob_dist = preds.probabilities[0]
        self.assertEqual(len(first_prob_dist), self.num_classes)
        self.assertAlmostEqual(sum(first_prob_dist), 1.0, places=4)

    def test_engine_supports_batch_mapping(self) -> None:
        """Verify compatibility with the mapping returned by the custom loader."""
        batch_dataset = MockBatchDataset(self.features, self.labels)
        batch_loader = DataLoader(
            batch_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=batch_collate,
        )

        # Train loop compatibility
        train_result = train_one_epoch(
            model=self.model,
            loader=batch_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
        )
        self.assertEqual(train_result.num_samples, self.num_samples)

        # Evaluation loop compatibility
        eval_result = evaluate_epoch(
            model=self.model,
            loader=batch_loader,
            criterion=self.criterion,
            device=self.device,
        )
        self.assertEqual(eval_result.predictions.sample_ids[0], "sample_0")
        self.assertEqual(len(eval_result.predictions.sample_ids), self.num_samples)


if __name__ == "__main__":
    unittest.main()