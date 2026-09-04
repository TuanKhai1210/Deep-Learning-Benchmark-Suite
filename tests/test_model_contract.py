"""Enable these tests once models are implemented and ML dependencies installed.

Remove the skip decorator for completed functionality; do not turn real failures
into skips/xfails. These tests do not prove model quality or benchmark validity.
"""

from pathlib import Path
import unittest

from dlbench.common.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@unittest.skip("TODO A/B/C (Thiên/Khoa/Khải): implement all five models; install the ml extra; remove this skip.")
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
                before = [parameter.detach().clone() for parameter in model.parameters()]
                optimizer.zero_grad()
                logits = model(torch.randn(7, 1, 28, 28))
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(7))
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                optimizer.step()
                self.assertTrue(any(not torch.equal(old, new.detach())
                                    for old, new in zip(before, model.parameters())))
