import unittest

import torch
from torch import nn

from dlbench.a1.metrics import count_parameters
from dlbench.a1.models.rnn import RNNClassifier


class TestRNNClassifier(unittest.TestCase):
    @staticmethod
    def _parameters(**overrides: object) -> dict[str, object]:
        parameters: dict[str, object] = {
            "input_size": 28,
            "num_classes": 10,
            "cell": "lstm",
            "representation": "rows",
            "hidden_size": 32,
            "num_layers": 1,
            "bidirectional": False,
        }
        parameters.update(overrides)
        return parameters

    # ── Shape contract ──────────────────────────────────────────────

    def test_forward_accepts_image_batches_and_returns_logits(self) -> None:
        model = RNNClassifier(self._parameters()).eval()

        for batch_size in (1, 7):
            with self.subTest(batch_size=batch_size):
                images = torch.randn(batch_size, 1, 28, 28)
                logits = model(images)

                self.assertEqual(tuple(logits.shape), (batch_size, 10))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_output_shape_is_batch_by_num_classes(self) -> None:
        for num_classes in (2, 10):
            with self.subTest(num_classes=num_classes):
                model = RNNClassifier(self._parameters(num_classes=num_classes)).eval()
                logits = model(torch.randn(4, 1, 28, 28))
                self.assertEqual(logits.shape, (4, num_classes))

    # ── Cell types ──────────────────────────────────────────────────

    def test_lstm_cell_produces_finite_logits(self) -> None:
        model = RNNClassifier(self._parameters(cell="lstm")).eval()
        logits = model(torch.randn(3, 1, 28, 28))

        self.assertEqual(logits.shape, (3, 10))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_gru_cell_produces_finite_logits(self) -> None:
        model = RNNClassifier(self._parameters(cell="gru")).eval()
        logits = model(torch.randn(3, 1, 28, 28))

        self.assertEqual(logits.shape, (3, 10))
        self.assertTrue(torch.isfinite(logits).all().item())

    # ── Representation ──────────────────────────────────────────────

    def test_rows_and_columns_produce_different_outputs(self) -> None:
        """Scanning the image by rows vs columns should give different logits,
        confirming the permute actually changes the sequence order."""
        torch.manual_seed(42)
        rows_model = RNNClassifier(self._parameters(representation="rows"))

        # Copy the exact same weights into a columns model
        cols_model = RNNClassifier(self._parameters(representation="columns"))
        cols_model.load_state_dict(rows_model.state_dict())

        rows_model.eval()
        cols_model.eval()

        # Use an asymmetric image so rows != columns
        images = torch.randn(2, 1, 28, 28)

        with torch.inference_mode():
            rows_out = rows_model(images)
            cols_out = cols_model(images)

        self.assertFalse(torch.allclose(rows_out, cols_out),
                         "Rows and columns representations should produce different outputs")

    def test_columns_representation_produces_correct_shape(self) -> None:
        model = RNNClassifier(self._parameters(representation="columns")).eval()
        logits = model(torch.randn(4, 1, 28, 28))
        self.assertEqual(logits.shape, (4, 10))

    # ── Bidirectional ───────────────────────────────────────────────

    def test_bidirectional_lstm_shape(self) -> None:
        model = RNNClassifier(self._parameters(cell="lstm", bidirectional=True)).eval()
        logits = model(torch.randn(3, 1, 28, 28))
        self.assertEqual(logits.shape, (3, 10))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_bidirectional_gru_shape(self) -> None:
        model = RNNClassifier(self._parameters(cell="gru", bidirectional=True)).eval()
        logits = model(torch.randn(3, 1, 28, 28))
        self.assertEqual(logits.shape, (3, 10))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_bidirectional_doubles_fc_input_features(self) -> None:
        hidden_size = 16
        uni = RNNClassifier(self._parameters(hidden_size=hidden_size, bidirectional=False))
        bi = RNNClassifier(self._parameters(hidden_size=hidden_size, bidirectional=True))

        self.assertEqual(uni.fc.in_features, hidden_size)
        self.assertEqual(bi.fc.in_features, hidden_size * 2)

    # ── Multi-layer ─────────────────────────────────────────────────

    def test_multi_layer_lstm(self) -> None:
        model = RNNClassifier(self._parameters(cell="lstm", num_layers=3)).eval()
        logits = model(torch.randn(2, 1, 28, 28))
        self.assertEqual(logits.shape, (2, 10))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_multi_layer_bidirectional_gru(self) -> None:
        model = RNNClassifier(
            self._parameters(cell="gru", num_layers=2, bidirectional=True)
        ).eval()
        logits = model(torch.randn(4, 1, 28, 28))
        self.assertEqual(logits.shape, (4, 10))
        self.assertTrue(torch.isfinite(logits).all().item())

    # ── Backward pass ───────────────────────────────────────────────

    def test_backward_updates_parameters(self) -> None:
        model = RNNClassifier(self._parameters()).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images = torch.randn(7, 1, 28, 28)
        targets = torch.arange(7)
        before = [p.detach().clone() for p in model.parameters()]

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

    def test_backward_updates_parameters_bidirectional(self) -> None:
        model = RNNClassifier(self._parameters(bidirectional=True)).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images = torch.randn(4, 1, 28, 28)
        targets = torch.randint(0, 10, (4,))
        before = [p.detach().clone() for p in model.parameters()]

        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )

    # ── Parameter counting ──────────────────────────────────────────

    def test_all_parameters_are_trainable(self) -> None:
        model = RNNClassifier(self._parameters())
        counts = count_parameters(model)
        self.assertEqual(counts["total_parameters"], counts["trainable_parameters"])
        self.assertGreater(counts["total_parameters"], 0)

    def test_bidirectional_has_more_parameters(self) -> None:
        uni = RNNClassifier(self._parameters(hidden_size=16, bidirectional=False))
        bi = RNNClassifier(self._parameters(hidden_size=16, bidirectional=True))

        uni_count = count_parameters(uni)["total_parameters"]
        bi_count = count_parameters(bi)["total_parameters"]
        self.assertGreater(bi_count, uni_count)

    # ── Eval determinism ────────────────────────────────────────────

    def test_eval_mode_is_deterministic(self) -> None:
        model = RNNClassifier(self._parameters()).eval()
        images = torch.randn(5, 1, 28, 28)

        with torch.inference_mode():
            first = model(images)
            second = model(images)

        self.assertTrue(torch.equal(first, second))

    # ── Validation errors ───────────────────────────────────────────

    def test_missing_required_parameters_raise(self) -> None:
        required = self._parameters()
        for key in ("input_size", "num_classes", "cell", "representation", "hidden_size"):
            with self.subTest(missing=key):
                params = required.copy()
                del params[key]
                with self.assertRaises(ValueError):
                    RNNClassifier(params)

    def test_invalid_cell_type_raises(self) -> None:
        for cell in ("rnn", "vanilla", "transformer", ""):
            with self.subTest(cell=cell):
                with self.assertRaises(ValueError):
                    RNNClassifier(self._parameters(cell=cell))

    def test_invalid_representation_raises(self) -> None:
        for rep in ("patches", "pixels", "diagonal", ""):
            with self.subTest(representation=rep):
                with self.assertRaises(ValueError):
                    RNNClassifier(self._parameters(representation=rep))

    # ── All four cell × representation combos ───────────────────────

    def test_all_cell_representation_combinations(self) -> None:
        for cell in ("lstm", "gru"):
            for rep in ("rows", "columns"):
                for bidir in (False, True):
                    with self.subTest(cell=cell, representation=rep, bidirectional=bidir):
                        model = RNNClassifier(
                            self._parameters(cell=cell, representation=rep, bidirectional=bidir)
                        ).eval()
                        logits = model(torch.randn(2, 1, 28, 28))
                        self.assertEqual(logits.shape, (2, 10))
                        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
