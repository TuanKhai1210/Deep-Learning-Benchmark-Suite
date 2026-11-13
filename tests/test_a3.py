"""ABO-specific contracts and synthetic CLI integration checks."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dlbench.a3.cli import main
from dlbench.a3.data import CLASSES, SPLITS, assign_splits, load_manifest, validate_rows
from dlbench.a3.engine import classification_metrics, evaluate_test
from dlbench.a3.features import load_cache, mask_titles, save_cache
from dlbench.a3.model import MODALITIES


class A3Tests(unittest.TestCase):
    def test_group_split_is_repeatable_and_rejects_leakage(self):
        rows = [
            dict(sample_id=f"{label_index}-{i}", image_id=f"{label_index}-{i}",
                 image_path=f"{label_index}-{i}.jpg", title=f"Product {i}",
                 label=label, group_id=f"{label_index}-{i // 2}")
            for label_index, label in enumerate(CLASSES) for i in range(8)
        ]
        result = assign_splits(rows)
        self.assertEqual(result, assign_splits(list(reversed(rows))))
        for split in SPLITS:
            self.assertEqual({row["label"] for row in result if row["split"] == split}, set(CLASSES))
        result[0]["split"] = next(split for split in SPLITS if split != result[0]["split"])
        with self.assertRaisesRegex(ValueError, "across splits"):
            validate_rows(result)

    def test_cache_reorders_by_id_and_rejects_changed_manifest(self):
        rows = [
            dict(sample_id=f"s{i}", image_id=f"i{i}", image_path=f"{i}.jpg",
                 title="Test", label=CLASSES[i], group_id=f"g{i}", split="train")
            for i in range(3)
        ]
        vectors = np.repeat(np.arange(3, dtype=np.float32)[:, None], 512, axis=1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.npz"
            save_cache(path, rows[::-1], "image", vectors[::-1], {"model": "fixture"})
            aligned, _ = load_cache(path, rows, "image")
            torch.testing.assert_close(aligned, torch.from_numpy(vectors))
            rows[0]["title"] = "Changed"
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_cache(path, rows, "image")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                save_cache(path, [rows[0], rows[0], rows[2]], "image", vectors, {})

    def test_masking_and_macro_f1(self):
        self.assertEqual(
            mask_titles(["CHAIR and wall-art; lampshade", "Sneaker"], ["sneaker"]),
            ["[MASK] and [MASK]; lampshade", "[MASK]"],
        )
        target = torch.arange(len(CLASSES))
        prediction = target.clone()
        prediction[0] = 1
        metrics = classification_metrics(target, prediction)
        self.assertAlmostEqual(metrics["accuracy"], 0.9)
        self.assertAlmostEqual(metrics["macro_f1"], (8 + 2 / 3) / 10)
        self.assertEqual(metrics["confusion_matrix"][0][1], 1)

    def test_cli_smoke_and_holdout_provenance(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "smoke"
            main(["smoke", "--output", str(root)])
            expected = {row["sample_id"] for row in load_manifest(root / "manifest.csv")
                        if row["split"] == "test"}
            for variant in MODALITIES:
                result = root / variant / "test"
                with (result / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                    predictions = list(csv.DictReader(stream))
                self.assertEqual({row["sample_id"] for row in predictions}, expected)
                self.assertEqual(len(predictions), len(expected))
                metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), len(expected))
            product = json.loads((root / "product" / "run.json").read_text(encoding="utf-8"))
            mean = json.loads((root / "mean-ablation" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(product["trainable_head_parameters"], mean["trainable_head_parameters"])
            with np.load(root / "image.npz", allow_pickle=False) as cache:
                changed = {**cache, "features": cache["features"] + 1}
            np.savez(root / "changed.npz", **changed)
            with self.assertRaisesRegex(ValueError, "differ from the training run"):
                evaluate_test(root / "manifest.csv", {"image": root / "changed.npz"},
                              root / "image-only", torch.device("cpu"))
