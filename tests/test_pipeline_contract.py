"""Contract examples waiting for implementation, explicitly skipped for now."""

import unittest


class PipelineContractTests(unittest.TestCase):
    def test_metrics_use_all_ten_classes(self):
        from dlbench.a1.metrics import classification_metrics
        metrics = classification_metrics(list(range(10)), list(range(10)))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        metrics = classification_metrics(list(range(10)), [0] * 10)
        self.assertAlmostEqual(metrics["accuracy"], 0.1)
        self.assertAlmostEqual(metrics["macro_f1"], (2.0 / 11.0) / 10.0)

    def test_checkpoint_order_ignores_test_metrics(self):
        from dlbench.a1.checkpoint import is_better
        incumbent = {"val_macro_f1": 0.8, "val_loss": 0.5, "epoch": 4}
        candidate = {"val_macro_f1": 0.8, "val_loss": 0.4, "epoch": 5}
        self.assertTrue(is_better(candidate, incumbent))
        self.assertTrue(is_better({**incumbent, "epoch": 3}, incumbent))
        self.assertFalse(is_better({**incumbent, "test_accuracy": 1.0}, incumbent))
        self.assertTrue(is_better(incumbent, None))

    @unittest.skip("TODO A (Thiên): implement split validation; remove this skip.")
    def test_split_overlap_is_rejected(self):
        from dlbench.a1.contracts import SplitManifest
        from dlbench.a1.data.split import validate_split
        manifest = SplitManifest("fashion_mnist", 36, list(range(50000)),
                                 list(range(50000, 60000)), list(range(10000)))
        validate_split(manifest)
        # Replace one validation index with a training index; must fail.
        bad = SplitManifest("fashion_mnist", 36, manifest.train_indices,
                            [0] + manifest.validation_indices[1:], manifest.test_indices)
        with self.assertRaises(ValueError):
            validate_split(bad)
