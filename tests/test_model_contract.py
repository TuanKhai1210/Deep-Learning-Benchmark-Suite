"""Model contract tests; keep per-model tests active as models are completed.

The shared five-model checks remain skipped until every model is implemented.
These tests do not prove model quality or benchmark validity.
"""

import unittest
from pathlib import Path

from dlbench.common.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@unittest.skip(
    "TODO A/B/C (Thiên/Khoa/Khải): implement all five models; "
    "install the ml extra; remove this skip."
)
class ModelContractTests(unittest.TestCase):
    def test_all_models_accept_canonical_images(self):
        import torch
        from dlbench.a1.models.registry import build_model

        for path in (ROOT / "configs/a1/models").glob("*.py"):
            for batch_size in (1, 7):
                with self.subTest(model=path.stem, batch=batch_size):
                    model = build_model(load_config(path)["model"]).eval()
                    with torch.inference_mode():
                        logits = model(torch.randn(batch_size, 1, 28, 28))
                    self.assertEqual(tuple(logits.shape), (batch_size, 10))
                    self.assertTrue(torch.isfinite(logits).all().item())

    def test_all_models_update_a_parameter(self):
        import torch
        from dlbench.a1.models.registry import build_model

        for path in (ROOT / "configs/a1/models").glob("*.py"):
            with self.subTest(model=path.stem):
                model = build_model(load_config(path)["model"]).train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                before = [
                    parameter.detach().clone() for parameter in model.parameters()
                ]
                optimizer.zero_grad()
                logits = model(torch.randn(7, 1, 28, 28))
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(7))
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                self.assertTrue(
                    any(
                        not torch.equal(old, new.detach())
                        for old, new in zip(before, model.parameters())
                    )
                )


class TransformerContractTests(unittest.TestCase):
    def _build_model(self):
        from dlbench.a1.models.registry import build_model

        config = load_config(ROOT / "configs/a1/models/transformer.py")
        return build_model(config["model"])

    def test_output_shape_and_finite_logits(self):
        import torch

        model = self._build_model().eval()

        for batch_size in (1, 7):
            with self.subTest(batch_size=batch_size):
                with torch.inference_mode():
                    logits = model(torch.randn(batch_size, 1, 28, 28))

                self.assertEqual(tuple(logits.shape), (batch_size, 10))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_patchify_preserves_patch_values_and_order(self):
        import torch

        model = self._build_model()
        image = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
        patches = model.patchify(image)

        self.assertEqual(tuple(patches.shape), (1, 49, 16))
        for index, rows, columns in (
            (0, slice(0, 4), slice(0, 4)),
            (1, slice(0, 4), slice(4, 8)),
            (7, slice(4, 8), slice(0, 4)),
            (-1, slice(24, 28), slice(24, 28)),
        ):
            with self.subTest(patch=index):
                expected = image[0, 0, rows, columns].reshape(-1)
                self.assertTrue(torch.equal(patches[0, index], expected))

    def test_rejects_wrong_image_shape_and_dtype(self):
        import torch

        model = self._build_model()
        for shape in ((2, 3, 28, 28), (2, 1, 32, 32), (1, 28, 28)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    model(torch.randn(*shape))

        with self.assertRaises(TypeError):
            model(torch.randn(2, 1, 28, 28, dtype=torch.float64))

    def test_rejects_invalid_patch_or_head_configuration(self):
        from dlbench.a1.models.transformer import TransformerClassifier

        parameters = load_config(ROOT / "configs/a1/models/transformer.py")[
            "model"
        ]["parameters"]
        for overrides in ({"patch_size": 0}, {"patch_size": 3}, {"num_heads": 3}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    TransformerClassifier({**parameters, **overrides})

    def test_optimizer_step_updates_a_parameter(self):
        import torch

        model = self._build_model().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        optimizer.zero_grad()
        logits = model(torch.randn(7, 1, 28, 28))
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(7))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        for parameter in (
            model.patch_projection.weight,
            model.position_embedding,
            model.blocks[0].attention.q_proj.weight,
            model.classifier.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )

    def test_can_overfit_four_fixed_images(self):
        import torch
        from dlbench.a1.models.transformer import TransformerClassifier

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            parameters = dict(
                load_config(ROOT / "configs/a1/models/transformer.py")["model"][
                    "parameters"
                ]
            )
            parameters["dropout"] = 0.0
            model = TransformerClassifier(parameters).train()

            images = torch.randn(4, 1, 28, 28)
            labels = torch.arange(4)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

            with torch.no_grad():
                initial_loss = torch.nn.functional.cross_entropy(
                    model(images), labels
                ).item()

            for _ in range(20):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()

            with torch.inference_mode():
                logits = model(images)
                final_loss = torch.nn.functional.cross_entropy(
                    logits, labels
                ).item()

            self.assertLess(final_loss, initial_loss * 0.5)
            self.assertEqual((logits.argmax(dim=1) == labels).sum().item(), 4)
