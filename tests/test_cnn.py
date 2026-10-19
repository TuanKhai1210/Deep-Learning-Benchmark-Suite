import unittest

import torch
from torch import nn

from dlbench.a1.metrics import count_parameters
from dlbench.a1.models.cnn import CNNClassifier


class TestCNNClassifier(unittest.TestCase):
    @staticmethod
    def _parameters(**overrides: object) -> dict[str, object]:
        parameters: dict[str, object] = {
            "num_classes": 10,
            "channels": [8, 16],
            "hidden_dim": 32,
            "dropout": 0.0,
        }
        parameters.update(overrides)
        return parameters

    def test_initialization_from_project_config(self) -> None:
        from configs.a1.models.cnn import CONFIG

        model = CNNClassifier(CONFIG["model"]["parameters"])
        logits = model(torch.randn(4, 1, 28, 28))

        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_forward_accepts_image_batches_and_returns_logits(self) -> None:
        model = CNNClassifier(self._parameters()).eval()

        for batch_size in (1, 7):
            with self.subTest(batch_size=batch_size):
                logits = model(torch.randn(batch_size, 1, 28, 28))

                self.assertEqual(tuple(logits.shape), (batch_size, 10))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_output_shape_uses_num_classes(self) -> None:
        model = CNNClassifier(self._parameters(num_classes=3)).eval()

        logits = model(torch.randn(4, 1, 28, 28))

        self.assertEqual(tuple(logits.shape), (4, 3))

    def test_forward_rejects_invalid_image_shapes(self) -> None:
        model = CNNClassifier(self._parameters()).eval()

        for images in (
            torch.randn(1, 28, 28),
            torch.randn(4, 3, 28, 28),
            torch.randn(4, 1, 32, 32),
            torch.empty(0, 1, 28, 28),
        ):
            with self.subTest(input_shape=tuple(images.shape)):
                with self.assertRaises(ValueError):
                    model(images)

    def test_invalid_parameters_raise_value_error(self) -> None:
        invalid_parameters = (
            {"num_classes": 0},
            {"num_classes": True},
            {"channels": []},
            {"channels": [8, 0]},
            {"channels": [8, True]},
            {"hidden_dim": 0},
            {"hidden_dim": True},
            {"dropout": -0.1},
            {"dropout": 1.0},
            {"dropout": True},
        )

        for overrides in invalid_parameters:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    CNNClassifier(self._parameters(**overrides))

    def test_convolutional_and_classifier_layers_have_expected_dimensions(self) -> None:
        model = CNNClassifier(
            self._parameters(channels=[4, 8], hidden_dim=12, num_classes=3)
        )
        convolutions = [
            layer for layer in model.features if isinstance(layer, nn.Conv2d)
        ]

        self.assertEqual(
            [(layer.in_channels, layer.out_channels) for layer in convolutions],
            [(1, 4), (4, 8)],
        )
        self.assertEqual(model.classifier[1].in_features, 8)
        self.assertEqual(model.classifier[1].out_features, 12)
        self.assertEqual(model.classifier[-1].out_features, 3)

    def test_dropout_is_applied_only_in_training_mode(self) -> None:
        model = CNNClassifier(self._parameters(dropout=0.5))
        images = torch.ones(16, 1, 28, 28)

        model.eval()
        eval_first = model(images)
        eval_second = model(images)
        self.assertTrue(torch.equal(eval_first, eval_second))

        model.train()
        torch.manual_seed(0)
        train_first = model(images)
        train_second = model(images)
        self.assertFalse(torch.equal(train_first, train_second))

    def test_backward_updates_parameters(self) -> None:
        model = CNNClassifier(self._parameters()).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images = torch.randn(7, 1, 28, 28)
        targets = torch.arange(7)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )

    def test_all_parameters_are_trainable(self) -> None:
        model = CNNClassifier(self._parameters())

        counts = count_parameters(model)

        self.assertEqual(counts["total_parameters"], counts["trainable_parameters"])
        self.assertGreater(counts["total_parameters"], 0)


if __name__ == "__main__":
    unittest.main()
