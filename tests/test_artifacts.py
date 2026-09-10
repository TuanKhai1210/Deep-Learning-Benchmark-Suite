import tempfile
import unittest
from pathlib import Path

from copy import deepcopy
import json

from dlbench.common.config import load_config
from dlbench.common.artifacts import (
    create_run_dir,
    save_run_metadata,
)
ROOT = Path(__file__).resolve().parents[1]
class ArtifactTests(unittest.TestCase):
    def test_create_run_dir_creates_new_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs" / "a1"
            run_id = "fashion_mnist_linear_seed69420_run01"

            actual = create_run_dir(output_root, run_id)

            self.assertEqual(actual, output_root / run_id)
            self.assertTrue(actual.is_dir())
    def test_duplicate_run_preserves_existing_data(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs" / "a1"
            run_id = "mlp_seed69"
            run_dir = create_run_dir(output_root, run_id)

            marker = run_dir / "marker.txt"
            marker.write_text("old data", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_run_dir(output_root, run_id)

            self.assertEqual(
                marker.read_text(encoding="utf-8"),
                "old data",
            )
    def test_unsafe_run_ids_are_rejected(self):
        invalid_ids = [
            "", ".", "..",
            "../outside", "abc/def", r"abc\def",
            "mlp:69", "mlp run",
            "CON", "con", "NUL", "COM1", "LPT9",
        ]

        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"

            for run_id in invalid_ids:
                with self.subTest(run_id=run_id):
                    with self.assertRaises(ValueError):
                        create_run_dir(output_root, run_id)

            self.assertFalse(output_root.exists())
    def test_non_string_run_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"

            for run_id in [123, None, ["mlp_seed69"]]:
                with self.subTest(run_id=run_id):
                    with self.assertRaises(TypeError):
                        create_run_dir(output_root, run_id)

            self.assertFalse(output_root.exists())
    def test_valid_run_ids_are_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "runs"

            for run_id in ["mlp_seed69", "mlp-run01", "con_run69"]:
                with self.subTest(run_id=run_id):
                    result = create_run_dir(output_root, run_id)

                    self.assertEqual(result, output_root / run_id)
                    self.assertTrue(result.is_dir())
    def test_save_run_metadata_writes_reproducibility_artifacts(self):
        config = load_config(
            ROOT / "configs/a1/models/mlp.py"
        )

        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            split_path = temp_path / "split.json"

            split_path.write_text(
                json.dumps({
                    "split_seed": 36,
                    "train_indices": [0, 1],
                    "validation_indices": [2],
                }),
                encoding="utf-8",
            )

            resolved_config = deepcopy(config)
            resolved_config["data"]["split_file"] = str(split_path)
            resolved_config["preprocessing"]["mean"] = [0.0]
            resolved_config["preprocessing"]["std"] = [0.25]
            resolved_config["run"]["mode"] = "smoke"

            run_dir = create_run_dir(
                temp_path / "runs",
                "mlp_seed69420_smoke",
            )

            save_run_metadata(run_dir, resolved_config)

            expected_files = {
                "config.json",
                "environment.json",
                "metadata.json",
                "sources/model_config.py",
                "sources/protocol_config.py",
            }
            actual_files = {
                path.relative_to(run_dir).as_posix()
                for path in run_dir.rglob("*")
                if path.is_file()
            }

            self.assertEqual(actual_files, expected_files)

            saved_config = json.loads(
                (run_dir / "config.json").read_text(
                    encoding="utf-8"
                )
            )
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(saved_config, resolved_config)
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["run_mode"], "smoke")
            self.assertEqual(metadata["run_seed"], 69420)
            self.assertEqual(len(metadata["split"]["sha256"]), 64)
            self.assertEqual(
                len(metadata["normalization"]["sha256"]),
                64,
            )

            for name, source_location in config["_sources"].items():
                snapshot = run_dir / metadata["sources"][name]["snapshot"]

                self.assertEqual(
                    snapshot.read_bytes(),
                    Path(source_location).read_bytes(),
                )