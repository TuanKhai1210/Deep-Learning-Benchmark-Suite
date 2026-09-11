"""Configuration and CLI tests: no ML dependencies or dataset downloads required."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

try:
    import numpy  # type: ignore # pragma: no cover
except ModuleNotFoundError:  # CI does not install optional ML extras for config-only tests.
    class _FakeArray(list):
        def __init__(self, value):
            super().__init__(value)
            self.shape = self._infer_shape(value)

        @staticmethod
        def _infer_shape(value):
            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    return (len(value), len(value[0]))
                return (len(value),)
            return ()

        def __getitem__(self, key):
            if isinstance(key, list):
                return _FakeArray([self[i] for i in key])
            return super().__getitem__(key)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.array = lambda value, *args, **kwargs: _FakeArray(value) if not isinstance(value, _FakeArray) else value
    fake_numpy.arange = lambda *args, **kwargs: list(range(*args))
    fake_numpy.zeros = lambda shape, dtype=None: _FakeArray([[0 for _ in range(shape[1])] for _ in range(shape[0])]) if isinstance(shape, tuple) and len(shape) == 2 else _FakeArray([0 for _ in range(shape[0])])
    fake_numpy.uint8 = "uint8"
    sys.modules["numpy"] = fake_numpy

from dlbench.a1.cli import main
from dlbench.a1.contracts import EpochMetrics, Predictions
from dlbench.a1.models.registry import MODEL_REGISTRY, build_model
from dlbench.common.config import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "a1" / "models"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIGS / "linear.py")

    def test_all_model_configs_load_and_report_draft(self):
        files = sorted(CONFIGS.glob("*.py"))
        self.assertEqual(len(files), 5)
        for path in files:
            with self.subTest(model=path.stem):
                config = load_config(path)
                self.assertEqual(config["model"]["name"], path.stem)
                self.assertTrue(validate_config(config))
                self.assertEqual(config["data"], self.config["data"])

    def test_agreed_run_seeds_are_separate_from_split_seed(self):
        self.assertEqual(self.config["budget"]["run_seeds"], [69420, 67, 69])
        self.assertEqual(self.config["run"]["seed"], 69420)
        self.assertEqual(self.config["data"]["split_seed"], 36)

    def test_python_config_does_not_execute_statements(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unsafe.py"
            marker = Path(temp) / "must_not_exist.txt"
            path.write_text(
                f'from pathlib import Path\nPath({str(marker)!r}).touch()\nCONFIG = {{}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "no imports or executable statements"):
                load_config(path)
            self.assertFalse(marker.exists())

    def test_computed_config_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "computed.py"
            path.write_text('CONFIG = {"value": list(range(10))}\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "literal data"):
                load_config(path)

    def test_non_dictionary_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.py"
            path.write_text('CONFIG = []\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "dictionary"):
                load_config(path)

    def test_invalid_python_has_clear_config_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.py"
            path.write_text('CONFIG = {\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "Invalid Python configuration"):
                load_config(path)

    def test_draft_strict_check_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "Not ready"):
            validate_config(self.config, strict=True)

    def test_resolved_example_passes_config_only_gate(self):
        # Synthetic fixture for validator testing, NOT measured/approved data.
        config = deepcopy(self.config)
        config["protocol"].update(status="frozen", approved_by=["Example A", "Example B", "Example C"])
        config["preprocessing"].update(mean=[0.5], std=[0.25])
        config["budget"].update(max_epochs=2, tuning_trials_per_model=1)
        config["training"]["batch_size"] = 8
        config["timing"].update(device="cpu", batch_size=8)
        self.assertEqual(validate_config(config, strict=True), [])

    def test_model_cannot_override_shared_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.py"
            path.write_text('CONFIG = {"protocol_file": "../protocol.py", "data": {"split_seed": 99}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "override common"):
                load_config(path)

    def test_missing_model_table_is_rejected(self):
        del self.config["model"]
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_original_test_must_be_retained(self):
        self.config["data"]["test_size"] = 5000
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_unsupported_dataset_is_rejected(self):
        self.config["data"]["dataset"] = "mnist"
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_run_seed_must_be_common(self):
        self.config["run"]["seed"] = 99
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_invalid_model_name_type_is_a_config_error(self):
        self.config["model"]["name"] = []
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_current_split_contract_requires_stratification(self):
        self.config["data"]["stratified"] = False
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_loader_uses_nested_training_and_run_config(self):
        from dlbench.a1.data.loaders import build_dataloaders

        config = {
            "data": {"root": "./data", "split_file": "configs/a1/splits/fashion_mnist_seed36.json"},
            "training": {"batch_size": 32},
            "run": {"seed": 123},
            "preprocessing": {"image_size": [28, 28], "channels": 1},
        }

        class FakeManifest:
            train_indices = list(range(10))
            validation_indices = list(range(10, 20))
            test_indices = list(range(20, 30))

        with patch("dlbench.a1.data.loaders.load_split", return_value=FakeManifest()), \
             patch("dlbench.a1.data.loaders.load_official_dataset", return_value=object()), \
             patch("dlbench.a1.data.loaders.build_transforms", return_value="transform"), \
             patch("dlbench.a1.data.loaders.FashionMNISTSubset", return_value=object()), \
             patch("dlbench.a1.data.loaders.DataLoader") as mock_loader:
            build_dataloaders(config)

        train_call = mock_loader.call_args_list[0].kwargs
        self.assertEqual(train_call["batch_size"], 32)
        self.assertEqual(train_call["generator"].initial_seed(), 123)

    def test_eda_summary_tracks_split_metadata(self):
        from dlbench.a1.data.eda import generate_eda

        config = {
            "data": {"root": "./data", "split_seed": 36, "split_file": "configs/a1/splits/fashion_mnist_seed36.json"},
            "preprocessing": {"image_size": [28, 28], "channels": 1},
        }

        class FakeDataset:
            classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
            targets = list(range(10)) * 3

            def __getitem__(self, index):
                image = [[0 for _ in range(28)] for _ in range(28)]
                return (image, self.targets[index])

            def __len__(self):
                return len(self.targets)

        manifest = type("Manifest", (), {
            "train_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "validation_indices": [10, 11, 12],
            "test_indices": [13, 14, 15],
        })()

        with patch("dlbench.a1.data.eda.load_official_dataset", side_effect=[FakeDataset(), FakeDataset()]), \
             patch("dlbench.a1.data.eda.load_split", return_value=manifest), \
             tempfile.TemporaryDirectory() as tempdir:
            generate_eda(config, Path(tempdir))
            summary = json.loads(Path(tempdir, "eda_summary.json").read_text(encoding="utf-8"))

        self.assertIn("split", summary)
        self.assertEqual(summary["split"]["seed"], 36)
        self.assertEqual(summary["split"]["file"], "configs/a1/splits/fashion_mnist_seed36.json")

    def test_source_paths_are_retained_for_run_metadata(self):
        self.assertEqual(Path(self.config["_sources"]["model_config"]), CONFIGS / "linear.py")
        self.assertTrue(Path(self.config["_sources"]["protocol_config"]).is_file())

    def test_duplicate_seeds_are_rejected(self):
        self.config["budget"]["run_seeds"] = [69420, 69420]
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_invalid_std_is_rejected(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(std=invalid):
                config = deepcopy(self.config)
                config["preprocessing"].update(mean=[0.5], std=[invalid])
                with self.assertRaises(ConfigError):
                    validate_config(config)

    def test_negative_budget_is_rejected(self):
        self.config["budget"]["max_epochs"] = -2
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_wrong_label_order_is_rejected(self):
        self.config["evaluation"]["labels"].reverse()
        with self.assertRaises(ConfigError):
            validate_config(self.config)

    def test_each_load_is_independent(self):
        self.config["budget"]["run_seeds"].append(99)
        fresh = load_config(CONFIGS / "linear.py")
        self.assertEqual(fresh["budget"]["run_seeds"], [69420, 67, 69])

    def test_registry_matches_config_files(self):
        self.assertEqual(set(MODEL_REGISTRY), {p.stem for p in CONFIGS.glob("*.py")})

    def test_unknown_model_rejected_without_torch(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model({"name": "not_a_model", "parameters": {}})
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model({"name": [], "parameters": {}})

    def test_contracts_import_without_torch(self):
        metrics = EpochMetrics(loss=1.0, accuracy=0.5, macro_f1=0.4, num_samples=2)
        predictions = Predictions(["example"], [0], [0], [[1.0] + [0.0] * 9])
        self.assertEqual(metrics.num_samples, 2)
        self.assertEqual(len(predictions.probabilities[0]), 10)


class CLITests(unittest.TestCase):
    def test_each_agreed_seed_overrides_run_without_resplitting(self):
        for seed in (69420, 67, 69):
            with self.subTest(seed=seed), redirect_stdout(io.StringIO()) as output:
                code = main(["validate-config", "--config", str(CONFIGS / "linear.py"),
                             "--seed", str(seed)])
                self.assertEqual(code, 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["run_seed"], seed)
                self.assertEqual(result["split_seed"], 36)
        self.assertEqual(load_config(CONFIGS / "linear.py")["run"]["seed"], 69420)

    def test_cli_rejects_unapproved_run_seed(self):
        with redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                main(["validate-config", "--config", str(CONFIGS / "linear.py"), "--seed", "36"])
        self.assertEqual(result.exception.code, 2)
        self.assertIn("run.seed must belong", output.getvalue())

    def test_help_exits_successfully(self):
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                main(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("validate-config", output.getvalue())

    def test_validate_command_reports_unresolved(self):
        with redirect_stdout(io.StringIO()) as output:
            code = main(["validate-config", "--config", str(CONFIGS / "linear.py")])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.getvalue())["config_decisions_ready"])

    def test_strict_command_rejects_draft(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["validate-config", "--config", str(CONFIGS / "linear.py"), "--strict"])
        self.assertEqual(result.exception.code, 2)

    def test_train_does_not_start_on_draft(self):
        with redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                main(["train", "--config", str(CONFIGS / "linear.py")])
        self.assertEqual(result.exception.code, 2)
        self.assertIn("Not ready", output.getvalue())

    def test_prepare_success_path_mocked(self):
        import sys
        mock_dataset_mod = MagicMock()
        mock_metadata = {
            "dataset": "FashionMNIST",
            "split_seed": 36,
            "split_file": "configs/a1/splits/fashion_mnist_split.json",
            "num_train": 50000,
            "num_val": 10000,
            "num_test": 10000,
            "measured_mean": [0.2858],
            "measured_std": [0.3527],
            "classes": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        }
        mock_dataset_mod.prepare_data.return_value = mock_metadata
        
        with patch.dict(sys.modules, {"dlbench.a1.data.dataset": mock_dataset_mod}):
            with redirect_stdout(io.StringIO()) as output:
                code = main(["prepare", "--config", str(CONFIGS / "linear.py")])
                self.assertEqual(code, 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["dataset"], "FashionMNIST")


if __name__ == "__main__":
    unittest.main()
