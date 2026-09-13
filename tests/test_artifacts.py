import tempfile
import csv
import hashlib
import unittest
from pathlib import Path

from copy import deepcopy
import json

from dlbench.common.config import load_config
from dlbench.common.artifacts import (
    CHECKPOINT_PROVENANCE_KEYS,
    checkpoint_provenance,
    create_run_dir,
    save_run_metadata,
    append_history,
    save_metrics,
    compute_data_provenance,
)
ROOT = Path(__file__).resolve().parents[1]
class ArtifactTests(unittest.TestCase):
    @staticmethod
    def history_row(epoch=0):
        return dict(epoch=epoch, train_loss=0.8, val_loss=0.9,
                    train_accuracy=0.7, val_accuracy=0.6,
                    train_macro_f1=0.65, val_macro_f1=0.55,
                    learning_rate=0.001, epoch_seconds=1.2)

    def test_history_appends_without_duplicate_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            append_history(root, self.history_row())
            append_history(root, self.history_row(1))
            with (root / "history.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["epoch"] for r in rows], ["0", "1"])

    def test_history_rejections_preserve_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            append_history(root, self.history_row())
            before = (root / "history.csv").read_bytes()
            bad_rows = [self.history_row(), {**self.history_row(1), "val_loss": float("nan")},
                        {**self.history_row(1), "val_accuracy": 1.1},
                        {**self.history_row(1), "epoch": True},
                        {**self.history_row(1), "extra": 1}]
            for row in bad_rows:
                with self.subTest(row=row), self.assertRaises(ValueError):
                    append_history(root, row)
                self.assertEqual((root / "history.csv").read_bytes(), before)

    def test_history_rejects_wrong_existing_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "history.csv"
            path.write_text("epoch,wrong\n0,1\n", encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                append_history(root, self.history_row(1))
            self.assertEqual(path.read_bytes(), before)

    def test_history_rejects_corrupt_existing_rows_without_modification(self):
        corrupt_rows = {
            "non_numeric": b"1,broken,0.9,0.7,0.6,0.65,0.55,0.001,1.2\n",
            "missing_column": b"1,0.8,0.9,0.7,0.6,0.65,0.55,0.001\n",
            "extra_column": b"1,0.8,0.9,0.7,0.6,0.65,0.55,0.001,1.2,extra\n",
            "non_finite": b"1,nan,0.9,0.7,0.6,0.65,0.55,0.001,1.2\n",
        }
        for name, damaged_row in corrupt_rows.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                append_history(root, self.history_row())
                path = root / "history.csv"
                with path.open("ab") as file:
                    file.write(damaged_row)
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "invalid row"):
                    append_history(root, self.history_row(2))
                self.assertEqual(path.read_bytes(), before)

    def test_history_rejects_partial_final_line_without_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            append_history(root, self.history_row())
            path = root / "history.csv"
            with path.open("ab") as file:
                file.write(b"1,0.8,0.9,0.7")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "incomplete final line"):
                append_history(root, self.history_row(2))
            self.assertEqual(path.read_bytes(), before)

    @staticmethod
    def metrics_payload():
        return dict(eval_split="validation", epoch=0, val_loss=0.5,
                    val_accuracy=0.8, val_macro_f1=0.7,
                    timing_scope="fit", timing_units="seconds")

    def test_metrics_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = self.metrics_payload()
            save_metrics(root, payload)
            path = root / "metrics.json"
            self.assertEqual(json.loads(path.read_text()), payload)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_metrics(root, payload)
            self.assertEqual(path.read_bytes(), before)
            test = {k.replace("val_", "test_"): v for k, v in payload.items()}
            test["eval_split"] = "test"
            save_metrics(root, test)
            self.assertEqual(json.loads((root / "metrics_test.json").read_text()), test)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_metrics_do_not_create_file(self):
        for change in ({"val_loss": float("nan")}, {"epoch": -1},
                       {"val_accuracy": 2}, {"test_loss": 1},
                       {"extra": Path("unsupported")}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises((ValueError, TypeError)):
                    save_metrics(Path(temp), {**self.metrics_payload(), **change})
                self.assertEqual(list(Path(temp).iterdir()), [])

    def test_data_provenance_needs_no_run_or_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.json"
            path.write_bytes(b'{"train_indices": [0]}')
            config = {"data": {"split_file": str(path)},
                      "preprocessing": {"mean": [0.1], "std": [0.2]}}
            first = compute_data_provenance(config)
            self.assertEqual(first["split_hash"], hashlib.sha256(path.read_bytes()).hexdigest())
            config["preprocessing"] = {"std": [0.2], "mean": [0.1]}
            self.assertEqual(first, compute_data_provenance(config))
            path.write_bytes(b'{"train_indices": [1]}')
            self.assertNotEqual(first["split_hash"], compute_data_provenance(config)["split_hash"])
            config["preprocessing"]["std"] = [0.3]
            self.assertNotEqual(first["statistics_hash"], compute_data_provenance(config)["statistics_hash"])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                compute_data_provenance(config)

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

            returned_metadata = save_run_metadata(
                run_dir,
                resolved_config,
            )

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
            self.assertEqual(returned_metadata, metadata)
            self.assertEqual(compute_data_provenance(resolved_config), {
                "split_hash": metadata["split"]["sha256"],
                "statistics_hash": metadata["normalization"]["sha256"],
            })
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["run_mode"], "smoke")
            self.assertEqual(metadata["run_seed"], 69420)
            self.assertEqual(len(metadata["split"]["sha256"]), 64)
            self.assertEqual(
                len(metadata["normalization"]["sha256"]),
                64,
            )

            provenance = checkpoint_provenance(returned_metadata)
            self.assertEqual(
                set(provenance),
                CHECKPOINT_PROVENANCE_KEYS,
            )
            self.assertEqual(
                provenance["split_hash"],
                metadata["split"]["sha256"],
            )
            self.assertEqual(
                provenance["statistics_hash"],
                metadata["normalization"]["sha256"],
            )

            for name, source_location in config["_sources"].items():
                snapshot = run_dir / metadata["sources"][name]["snapshot"]

                self.assertEqual(
                    snapshot.read_bytes(),
                    Path(source_location).read_bytes(),
                )
