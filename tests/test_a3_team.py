"""Regression coverage for reviewed ABO grouping and the five-model workflow."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dlbench.a3.analysis import analyze
from dlbench.a3.cli import main
from dlbench.a3.curation import DECISION_FIELDS, audit_candidates, freeze_manifest, verified_freeze
from dlbench.a3.data import CLASSES, FIELDS, SPLITS, read_csv, write_csv
from dlbench.a3.features import WIDTHS, inspect_title_policy, save_cache
from dlbench.a3.model import MODALITIES
from dlbench.a3.team_batches import TEAM_SEEDS, collect_batches, lock_benchmark, train_batch


class TeamWorkflowTests(unittest.TestCase):
    def test_reviewed_groups_and_image_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_root = root / "images"
            image_root.mkdir()
            rows = []
            rng = np.random.default_rng(36)
            for index, label in enumerate(CLASSES):
                for sample in range(5):
                    key = f"{index}-{sample}"
                    rows.append(dict(sample_id=key, image_id=key, image_path=key + ".png",
                                     title=label, label=label, group_id=key))
                    if key not in {"0-0", "1-0"}:
                        Image.fromarray(rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)).save(
                            image_root / (key + ".png"))
            (image_root / "0-0.png").write_bytes((image_root / "0-1.png").read_bytes())
            (image_root / "1-0.png").write_bytes(b"broken")
            candidates = root / "candidates.csv"
            write_csv(candidates, FIELDS, rows)
            decisions = audit_candidates(candidates, image_root, root / "audit",
                                         sample_per_class=2) / "decisions.csv"
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                freeze_manifest(decisions.parent, image_root, root / "cohort", reviewer="Fixture")
            edits = read_csv(decisions)
            for row in edits:
                row["decision"] = "separate" if row["related_id"] else "keep"
            edits.append(dict(sample_id="0-1", related_id="0-2", reason="manual",
                              decision="merge", note="Fixture variant"))
            edits.extend([
                dict(sample_id="0-0", related_id="0-2", reason="manual", decision="keep", note="Linked through 0-1"),
                dict(sample_id="2-0", related_id="2-1", reason="manual", decision="keep", note="No other link"),
            ])
            with decisions.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
                writer.writeheader()
                writer.writerows(edits)
            manifest = freeze_manifest(decisions.parent, image_root, root / "cohort", reviewer="Fixture")
            sealed = verified_freeze(manifest.parent, image_root=image_root)
            ids = {row["sample_id"]: row for row in sealed}
            self.assertNotIn("1-0", ids)
            self.assertNotEqual(ids["2-0"]["group_id"], ids["2-1"]["group_id"])
            self.assertEqual(len({ids[f"0-{index}"]["group_id"] for index in range(3)}), 1)
            self.assertEqual(len({ids[f"0-{index}"]["split"] for index in range(3)}), 1)
            (image_root / "0-0.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen image file changed"):
                verified_freeze(manifest.parent, image_root=image_root)

    def test_team_cli_locks_validation_and_analysis_rejects_changes(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                dict(sample_id=f"{label}-{split}", image_id=f"{label}-{split}",
                     image_path=f"{label.lower()}-{split}.jpg", title=label + " " + split,
                     label=label, group_id=f"{label}-{split}", split=split)
                for label in CLASSES for split in SPLITS
            ]
            self.assertEqual(inspect_title_policy(rows, ["train"])["train_support"], {"train": 10})
            with self.assertRaisesRegex(ValueError, "absent from train"):
                inspect_title_policy(rows, ["test"])
            manifest = root / "manifest.csv"
            write_csv(manifest, (*FIELDS, "split"), rows)
            rng = np.random.default_rng(36)
            args = ["--manifest", str(manifest), "--device", "cpu", "--output", str(root / "suite")]
            for modality, width in WIDTHS.items():
                cache = root / f"{modality}.npz"
                save_cache(cache, rows, modality, rng.standard_normal((len(rows), width), dtype=np.float32), {})
                args.extend([f"--{modality}-cache", str(cache)])
            config = root / "config.py"
            config.write_text("CONFIG = {'batch_size': 5, 'epochs': 2, 'projection': 8, 'learning_rate': .001}",
                              encoding="utf-8")
            main(["train-team", *args, "--config", str(config), "--seeds", "69420", "67", "69"])
            suite = root / "suite"
            plan = json.loads((suite / "suite.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["runs"]), 15)
            self.assertFalse(list(suite.glob("*/test")))
            main(["test-team", *args])
            result = json.loads((suite / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(result["paired"]["concat_vs_best_unimodal"]["right"],
                             plan["unimodal_reference"]["model"])
            output = analyze(suite, root / "analysis")
            self.assertEqual(len(read_csv(output / "per_class.csv")), 50)
            self.assertEqual(len(read_csv(output / "paired_deltas.csv")), 9)
            predictions = suite / "image-only-67" / "test" / "predictions.csv"
            with predictions.open("a", encoding="utf-8") as stream:
                stream.write("unexpected,SHOES,CHAIR\n")
            with self.assertRaisesRegex(ValueError, "membership"):
                analyze(suite, root / "changed-analysis")


    def test_distributed_batches_lock_inputs_and_resume_completed_runs(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                dict(sample_id=f"{label}-{split}", image_id=f"{label}-{split}",
                     image_path=f"{label.lower()}-{split}.jpg", title=label + " " + split,
                     label=label, group_id=f"{label}-{split}", split=split)
                for label in CLASSES for split in SPLITS
            ]
            manifest = root / "manifest.csv"
            write_csv(manifest, (*FIELDS, "split"), rows)
            rng = np.random.default_rng(67)
            caches = {name: root / f"{name}.npz" for name in WIDTHS}
            for name, width in WIDTHS.items():
                save_cache(caches[name], rows, name,
                           rng.standard_normal((len(rows), width), dtype=np.float32), {})
            config = {"batch_size": 8, "epochs": 2, "projection": 8, "learning_rate": .001}
            config_path = root / "head.py"
            config_path.write_text(f"CONFIG = {config!r}\n", encoding="utf-8")
            lock = root / "benchmark_lock.json"
            inputs = (manifest, caches, config_path, config, lock)
            lock_benchmark(*inputs)
            lock_benchmark(*inputs)
            device = torch.device("cpu")
            batches = [root / f"worker-{i}" for i in range(3)]
            subset = [("image-only", 67), ("image-only", 69)]
            other = [(model, seed) for seed in TEAM_SEEDS for model in MODALITIES
                     if (model, seed) not in subset]
            output = root / "suite"
            train_batch(*inputs, subset, batches[0], device)
            with patch("dlbench.a3.team_batches.train", side_effect=AssertionError("Already complete")):
                train_batch(*inputs, subset, batches[0], device)
            with self.assertRaisesRegex(ValueError, "Missing completed runs:.*product"):
                collect_batches(*inputs, batches[:1], output)
            with self.assertRaisesRegex(ValueError, "Duplicate completed run"):
                collect_batches(*inputs, [batches[0], batches[0]], output)
            self.assertFalse(output.exists())

            config_source = config_path.read_text(encoding="utf-8")
            config_path.write_text(config_source + "# edited\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                train_batch(*inputs, other[:5], batches[1], device)
            config_path.write_text(config_source, encoding="utf-8")
            train_batch(*inputs, other[:5], batches[1], device)
            train_batch(*inputs, other[5:], batches[2], device)
            plan = json.loads(collect_batches(*inputs, batches, output).read_text(encoding="utf-8"))
            self.assertEqual(len(plan["runs"]), 15)
            self.assertEqual(len({(run["model"], run["seed"]) for run in plan["runs"]}), 15)
            self.assertEqual(sum(item["batch_id"] == "worker-0" for item in plan["runs"]), 2)
            self.assertFalse(list(output.glob("*/test")))
            args = ["--manifest", str(manifest), "--device", "cpu", "--output", str(output)]
            for modality, cache in caches.items():
                args.extend([f"--{modality}-cache", str(cache)])
            main(["test-team", *args])
            analysis = analyze(output, root / "analysis")
            scores = read_csv(analysis / "scores_by_seed.csv")
            self.assertEqual(len(scores), 15)
            self.assertEqual({row["training_device"] for row in scores}, {"cpu"})
            summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
            self.assertNotIn("fit_loop_seconds", summary["by_model"]["image-only"])
