"""Synthetic artifacts and CPU-only timing; no data download or training."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dlbench.a1.analysis import analyze_run, analyze_predictions, compare_runs, save_predictions
from dlbench.a1.benchmark import benchmark_inference
from dlbench.a1.contracts import Predictions
from dlbench.common.artifacts import HISTORY_COLUMNS


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.run.mkdir()
        config = {"model": {"name": "linear"}, "run": {"seed": 69},
                  "preprocessing": {}, "evaluation": {}, "timing": {}}
        metadata = {"model_name": "linear", "run_seed": 69, "run_mode": "main",
                    "git_revision": "abc", "git_dirty": False, "protocol_id": "a1-v0",
                    "split": {"sha256": "split"}, "normalization": {"sha256": "stats"}, "sources": {}}
        metrics = {"eval_split": "validation", "epoch": 0, "val_loss": 0.5,
                   "val_accuracy": 0.8, "val_macro_f1": 0.7}
        for name, payload in (("config", config), ("metadata", metadata), ("metrics", metrics)):
            (self.run / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        with (self.run / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(HISTORY_COLUMNS)
            writer.writerow([0, 0.4, 0.5, 0.9, 0.8, 0.8, 0.7, 0.001, 2])

    def test_curves_preserve_original_and_report_validation(self):
        original = (self.run / "metrics.json").read_bytes()
        analyze_run(self.run)
        self.assertEqual((self.run / "metrics.json").read_bytes(), original)
        summary = json.loads((self.run / "analysis/summary.json").read_text())
        self.assertEqual(summary["eval_split"], "validation")
        self.assertEqual(summary["completed_epochs"], 1)
        self.assertGreater((self.run / "analysis/learning_curves.png").stat().st_size, 100)
        with self.assertRaises(FileExistsError):
            analyze_run(self.run)

    def test_incomplete_history_rejected_before_output(self):
        path = self.run / "history.csv"
        path.write_text(path.read_text().rstrip(), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze_run(self.run)
        self.assertFalse((self.run / "analysis").exists())

    def test_wrong_best_metrics_rejected(self):
        path = self.run / "metrics.json"
        data = json.loads(path.read_text())
        data["val_macro_f1"] = 0.9
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "disagree"):
            analyze_run(self.run)

    def test_predictions_validate_before_creating_file(self):
        prediction = Predictions(["sample"], [0], [0], [[1.0] + [0.0] * 9])
        path = self.root / "predictions.csv"
        save_predictions(prediction, path)
        with self.assertRaises(FileExistsError):
            save_predictions(prediction, path)
        prediction.probabilities[0][0] = float("nan")
        with self.assertRaises(ValueError):
            save_predictions(prediction, self.root / "bad.csv")
        self.assertFalse((self.root / "bad.csv").exists())

    def test_comparison_rejects_duplicate_runs(self):
        with self.assertRaises(ValueError):
            compare_runs([self.run, self.run], self.root / "comparison.csv")

    def test_comparison_exports_validation_rows(self):
        compare_runs([self.run], self.root / "comparison.csv")
        self.assertIn("validation", (self.root / "comparison.csv").read_text())

    def test_confusion_keeps_absent_classes_and_sample_ids(self):
        predictions = Predictions(["image:1", "image:2"], [0, 1], [0, 0],
                                  [[1.0] + [0.0] * 9] * 2)
        analyze_predictions(predictions, self.root / "pred", split="validation")
        record = json.loads((self.root / "pred/errors.json").read_text())
        self.assertEqual(len(record["confusion_counts"]), 10)
        self.assertEqual(record["confusion_counts"][1][0], 1)
        self.assertEqual(record["errors"][0]["sample_id"], "image:2")
        self.assertEqual(sum(map(sum, record["confusion_counts"])), 2)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout())
        self.batch = torch.zeros(4, 3)
        self.config = {"device": "cpu", "batch_size": 4, "warmup_steps": 2,
                       "measurement_steps": 3, "precision": "float32", "scope": "forward_only"}

    def test_median_throughput_and_modes(self):
        self.model[1].eval()
        with patch("dlbench.a1.benchmark.time.perf_counter", side_effect=[0, .001, 1, 1.003, 2, 2.002]):
            result = benchmark_inference(self.model, self.batch, self.config)
        self.assertAlmostEqual(result["median_batch_ms"], 2)
        self.assertAlmostEqual(result["images_per_second"], 2000)
        self.assertEqual(result["parameter_count"], 8)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model[1].training)

    def test_wrong_batch_and_precision_rejected(self):
        for batch in (self.batch[:2], self.batch.double()):
            with self.assertRaises(ValueError):
                benchmark_inference(self.model, batch, self.config)

    def test_modes_restored_on_failure(self):
        with patch.object(self.model, "forward", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                benchmark_inference(self.model, self.batch, self.config)
        self.assertTrue(self.model.training)

    def test_runs_in_inference_eval_mode(self):
        seen = []
        hook = self.model.register_forward_pre_hook(
            lambda module, inputs: seen.append((module.training, torch.is_grad_enabled())))
        self.addCleanup(hook.remove)
        benchmark_inference(self.model, self.batch, self.config)
        self.assertEqual(seen, [(False, False)] * 5)


if __name__ == "__main__":
    unittest.main()
