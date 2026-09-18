import unittest

import torch
from torch import nn

from dlbench.a1.metrics import count_parameters
from dlbench.a1.models.linear import LinearClassifier


class TestLinearClassifier(unittest.TestCase):
    @staticmethod
    def _parameters(**overrides: object) -> dict[str, object]:
        parameters: dict[str, object] = {
            "input_dim": 784,
            "num_classes": 10,
        }
        parameters.update(overrides)
        return parameters

    def test_initialization_from_project_config(self) -> None:
        """Verify LinearClassifier builds cleanly from official config module."""
        from configs.a1.models.linear import CONFIG

        model = LinearClassifier(CONFIG["model"]["parameters"])
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_forward_accepts_image_batches_and_returns_logits(self) -> None:
        """Test shape contract for batch sizes 1 and 7."""
        model = LinearClassifier(self._parameters()).eval()

        for batch_size in (1, 7):
            with self.subTest(batch_size=batch_size):
                images = torch.randn(batch_size, 1, 28, 28)
                logits = model(images)

                self.assertEqual(tuple(logits.shape), (batch_size, 10))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_input_shape_polymorphism_and_flattening(self) -> None:
        """Verify model accepts both 4D [B, 1, 28, 28] and 2D [B, 784] tensors."""
        model = LinearClassifier(self._parameters()).eval()
        flat_images = torch.randn(3, 784)

        with torch.inference_mode():
            flat_logits = model(flat_images)
            image_logits = model(flat_images.reshape(3, 1, 28, 28))

        self.assertEqual(tuple(flat_logits.shape), (3, 10))
        self.assertTrue(torch.equal(flat_logits, image_logits))

    def test_forward_rejects_empty_batches(self) -> None:
        model = LinearClassifier(self._parameters()).eval()

        with self.assertRaises(ValueError):
            model(torch.empty(0, 1, 28, 28))

    def test_forward_rejects_input_without_batch_dimension(self) -> None:
        model = LinearClassifier(self._parameters()).eval()

        with self.assertRaises(ValueError):
            model(torch.randn(784))

    def test_parameter_count_matches_linear_weight_and_bias(self) -> None:
        model = LinearClassifier(self._parameters(input_dim=784, num_classes=10))

        # (784 * 10) weights + 10 biases = 7850
        self.assertEqual(
            count_parameters(model),
            {
                "total_parameters": 7850,
                "trainable_parameters": 7850,
            },
        )

    def test_gradient_update_modifies_parameters(self) -> None:
        """Verify backward pass and optimizer step update linear weights."""
        model = LinearClassifier(self._parameters()).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        before = [p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad()

        images = torch.randn(7, 1, 28, 28)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3, 4, 5, 6]))
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()
        optimizer.step()

        after = [p.detach().clone() for p in model.parameters()]
        self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
