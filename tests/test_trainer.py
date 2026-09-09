from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from dlbench.a1.contracts import EpochMetrics, EvaluationResult, Predictions
from dlbench.a1.trainer import (
    _checkpoint_metadata,
    _sha256_json,
    evaluate_checkpoint,
    fit,
    generate_run_id,
)


class TestTrainerHelpers(unittest.TestCase):
    def test_sha256_json_is_order_independent(self) -> None:
        left = {"b": 2, "a": [1, 2]}
        right = {"a": [1, 2], "b": 2}

        expected = hashlib.sha256(
            json.dumps(left, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        self.assertEqual(_sha256_json(left), expected)
        self.assertEqual(_sha256_json(left), _sha256_json(right))

    def test_checkpoint_metadata_hashes_split_and_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            split_path = Path(temporary_dir) / "split.json"
            split_contents = b'{"train": [0], "validation": [1]}'
            split_path.write_bytes(split_contents)
            config = {
                "data": {"split_file": str(split_path)},
                "preprocessing": {"mean": [0.1], "std": [0.2]},
                "git_revision": "abc123",
            }

            metadata = _checkpoint_metadata(config)

            self.assertEqual(metadata["split_hash"], hashlib.sha256(split_contents).hexdigest())
            self.assertEqual(metadata["statistics_hash"], _sha256_json(config["preprocessing"]))
            self.assertEqual(metadata["git_revision"], "abc123")

    def test_generate_run_id_contains_model_mode_seed_and_timestamp(self) -> None:
        run_id = generate_run_id(
            {"model": {"name": "mlp"}, "run": {"seed": 67}},
            smoke=True,
        )

        self.assertRegex(run_id, r"^a1_mlp_smoke_seed67_\d{8}-\d{6}_[0-9a-f]{6}$")


class TestFitCalculations(unittest.TestCase):
    @staticmethod
    def _config(output_root: str) -> dict[str, Any]:
        return {
            "model": {"name": "fake", "parameters": {}},
            "training": {
                "optimizer": "sgd",
                "learning_rate": 0.1,
                "weight_decay": 0.01,
                "momentum": 0.8,
            },
            "run": {"seed": 67, "output_root": output_root},
            "budget": {"max_epochs": 2},
            "data": {"split_file": "unused.json"},
            "preprocessing": {"mean": [0.1], "std": [0.2]},
        }

    def test_fit_records_metrics_and_selects_lower_loss_on_f1_tie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = self._config(temporary_dir)
            run_dir = Path(temporary_dir) / "run"
            model = torch.nn.Linear(1, 2)
            train_metrics = EpochMetrics(1.2, 0.4, 0.3, 4)
            validation_results = [
                EvaluationResult(
                    EpochMetrics(0.8, 0.5, 0.7, 4),
                    Predictions(["a"], [0], [0], [[1.0, 0.0]]),
                ),
                EvaluationResult(
                    EpochMetrics(0.6, 0.6, 0.7, 4),
                    Predictions(["a"], [0], [0], [[1.0, 0.0]]),
                ),
            ]
            saved_checkpoints: list[tuple[Path, dict, bool]] = []

            def record_checkpoint(path, payload, *, resume=False):
                saved_checkpoints.append((path, payload, resume))

            with patch("dlbench.a1.trainer.validate_config"), \
                    patch("dlbench.a1.trainer.seed_everything"), \
                    patch("dlbench.a1.trainer.create_run_dir", return_value=run_dir), \
                    patch("dlbench.a1.trainer.save_run_metadata"), \
                    patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                    patch("dlbench.a1.trainer.build_model", return_value=model), \
                    patch("dlbench.a1.trainer.train_one_epoch", return_value=train_metrics), \
                    patch("dlbench.a1.trainer.evaluate_epoch", side_effect=validation_results), \
                    patch("dlbench.a1.trainer.append_history") as append_history, \
                    patch("dlbench.a1.trainer.save_metrics") as save_metrics, \
                    patch("dlbench.a1.trainer.save_checkpoint", side_effect=record_checkpoint), \
                    patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                        "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                    }):
                result = fit(config)

            self.assertEqual(result.best_checkpoint, run_dir / "best.pt")
            self.assertEqual(append_history.call_count, 2)
            first_history = append_history.call_args_list[0].args[1]
            second_history = append_history.call_args_list[1].args[1]
            self.assertEqual(first_history["train_loss"], 1.2)
            self.assertEqual(first_history["val_loss"], 0.8)
            self.assertEqual(first_history["val_macro_f1"], 0.7)
            self.assertEqual(first_history["epoch"], 0)
            self.assertEqual(second_history["epoch"], 1)
            self.assertEqual(second_history["val_accuracy"], 0.6)
            self.assertEqual(second_history["learning_rate"], 0.1)
            self.assertEqual(save_metrics.call_args.args[1]["epoch"], 1)
            self.assertEqual(save_metrics.call_args.args[1]["val_loss"], 0.6)
            self.assertEqual(save_metrics.call_args.args[1]["val_accuracy"], 0.6)
            self.assertEqual(save_metrics.call_args.args[1]["val_macro_f1"], 0.7)
            best_saves = [item for item in saved_checkpoints if not item[2]]
            self.assertEqual([item[1]["epoch"] for item in best_saves], [0, 1])
            self.assertEqual(saved_checkpoints[-1][1]["val_metrics"], {
                "loss": 0.6,
                "accuracy": 0.6,
                "macro_f1": 0.7,
                "num_samples": 4,
            })

    def test_fit_uses_configured_sgd_hyperparameters(self) -> None:
        config = self._config("runs")
        captured_optimizers: list[torch.optim.Optimizer] = []
        train_metrics = EpochMetrics(1.0, 0.5, 0.4, 2)
        validation = EvaluationResult(
            EpochMetrics(0.9, 0.5, 0.4, 2),
            Predictions(["a"], [0], [0], [[1.0, 0.0]]),
        )

        def capture_optimizer(model, loader, optimizer, criterion, device):
            captured_optimizers.append(optimizer)
            return train_metrics

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                patch("dlbench.a1.trainer.train_one_epoch", side_effect=capture_optimizer), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=validation), \
                patch("dlbench.a1.trainer.append_history"), \
                patch("dlbench.a1.trainer.save_metrics"), \
                patch("dlbench.a1.trainer.save_checkpoint"), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config)

        self.assertEqual(len(captured_optimizers), 2)
        optimizer = captured_optimizers[0]
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 0.1)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)
        self.assertEqual(optimizer.defaults["momentum"], 0.8)

    def test_fit_records_decreasing_training_and_validation_loss(self) -> None:
        config = self._config("runs")
        features = torch.tensor([[-1.0], [1.0]] * 8)
        labels = torch.tensor([0, 1] * 8, dtype=torch.long)
        sample_ids = [f"sample-{index}" for index in range(len(labels))]
        dataset = [
            {"images": image, "labels": label, "sample_ids": sample_id}
            for image, label, sample_id in zip(features, labels, sample_ids)
        ]
        loader = DataLoader(cast(Any, dataset), batch_size=4, shuffle=False)
        model = torch.nn.Linear(1, 2)
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=loader, validation=loader)), \
                patch("dlbench.a1.trainer.build_model", return_value=model), \
                patch("dlbench.a1.trainer.append_history") as append_history, \
                patch("dlbench.a1.trainer.save_metrics") as save_metrics, \
                patch("dlbench.a1.trainer.save_checkpoint"), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config)

        history_rows = [call.args[1] for call in append_history.call_args_list]
        self.assertLess(history_rows[1]["train_loss"], history_rows[0]["train_loss"])
        self.assertLess(history_rows[1]["val_loss"], history_rows[0]["val_loss"])
        self.assertAlmostEqual(
            save_metrics.call_args.args[1]["val_loss"],
            history_rows[1]["val_loss"],
        )

    def test_fit_stops_after_configured_validation_patience(self) -> None:
        config = self._config("runs")
        config["budget"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 2
        config["training"]["min_delta"] = 0.01
        train_metrics = EpochMetrics(1.0, 0.5, 0.4, 2)
        validation_results = [
            EvaluationResult(
                EpochMetrics(1.0, 0.5, score, 2),
                Predictions(["a"], [0], [0], [[1.0, 0.0]]),
            )
            for score in (0.5, 0.5, 0.5)
        ]

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                patch("dlbench.a1.trainer.train_one_epoch", return_value=train_metrics), \
                patch("dlbench.a1.trainer.evaluate_epoch", side_effect=validation_results), \
                patch("dlbench.a1.trainer.append_history") as append_history, \
                patch("dlbench.a1.trainer.save_metrics"), \
                patch("dlbench.a1.trainer.save_checkpoint"), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config)

        self.assertEqual(append_history.call_count, 3)

    def test_fit_uses_smoke_budget_and_skips_main_run_strict_validation(self) -> None:
        config = self._config("runs")
        config["protocol"] = {"status": "draft", "id": "draft", "approved_by": ["A", "B", "C"]}
        config["budget"]["max_epochs"] = 25
        config["training"]["batch_size"] = 256
        config["training"]["save_frequency"] = 10
        train_metrics = EpochMetrics(1.0, 0.5, 0.4, 2)
        validation = EvaluationResult(
            EpochMetrics(1.0, 0.5, 0.4, 2),
            Predictions(["a"], [0], [0], [[1.0, 0.0]]),
        )

        with patch("dlbench.a1.trainer.validate_config") as validate_config, \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])) as build_dataloaders, \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                patch("dlbench.a1.trainer.train_one_epoch", return_value=train_metrics), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=validation), \
                patch("dlbench.a1.trainer.append_history"), \
                patch("dlbench.a1.trainer.save_metrics"), \
                patch("dlbench.a1.trainer.save_checkpoint"), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config, smoke=True)

        validate_config.assert_called_once()
        self.assertFalse(validate_config.call_args.kwargs["strict"])
        self.assertEqual(validate_config.call_args.args[0]["budget"]["max_epochs"], 2)
        self.assertEqual(validate_config.call_args.args[0]["training"]["batch_size"], 32)
        self.assertEqual(build_dataloaders.call_args.args[0]["budget"]["max_epochs"], 2)
        self.assertEqual(build_dataloaders.call_args.args[0]["training"]["batch_size"], 32)

    def test_fit_supports_validation_loss_as_early_stopping_monitor(self) -> None:
        config = self._config("runs")
        config["budget"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 2
        config["training"]["early_stopping_monitor"] = "loss"
        config["training"]["min_delta"] = 0.01
        train_metrics = EpochMetrics(1.0, 0.5, 0.4, 2)
        validation_results = [
            EvaluationResult(
                EpochMetrics(loss, 0.5, 0.4, 2),
                Predictions(["a"], [0], [0], [[1.0, 0.0]]),
            )
            for loss in (1.0, 1.0, 1.0)
        ]

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                patch("dlbench.a1.trainer.train_one_epoch", return_value=train_metrics), \
                patch("dlbench.a1.trainer.evaluate_epoch", side_effect=validation_results), \
                patch("dlbench.a1.trainer.append_history") as append_history, \
                patch("dlbench.a1.trainer.save_metrics"), \
                patch("dlbench.a1.trainer.save_checkpoint"), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config)

        self.assertEqual(append_history.call_count, 3)

    def test_fit_saves_last_checkpoint_when_early_stopping_interrupts_early(self) -> None:
        config = self._config("runs")
        config["budget"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 2
        config["training"]["save_frequency"] = 5
        train_metrics = EpochMetrics(1.0, 0.5, 0.4, 2)
        validation = EvaluationResult(
            EpochMetrics(1.0, 0.5, 0.4, 2),
            Predictions(["a"], [0], [0], [[1.0, 0.0]]),
        )
        saved_paths: list[tuple[Path, bool, int]] = []

        def record_checkpoint(path, payload, *, resume=False):
            saved_paths.append((path, resume, payload["epoch"]))

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                patch("dlbench.a1.trainer.train_one_epoch", return_value=train_metrics), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=validation), \
                patch("dlbench.a1.trainer.append_history"), \
                patch("dlbench.a1.trainer.save_metrics"), \
                patch("dlbench.a1.trainer.save_checkpoint", side_effect=record_checkpoint), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config)

        last_saves = [item for item in saved_paths if item[0].name == "last.pt"]
        self.assertTrue(last_saves)
        self.assertTrue(any(item[1] for item in last_saves))
        self.assertIn(2, [item[2] for item in last_saves])

    def test_fit_rejects_unknown_early_stopping_monitor(self) -> None:
        config = self._config("runs")
        config["training"]["early_stopping_monitor"] = "accuracy"

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)):
            with self.assertRaises(ValueError):
                fit(config)

    def test_fit_saves_last_checkpoint_at_frequency_and_final_epoch(self) -> None:
        config = self._config("runs")
        config["budget"]["max_epochs"] = 5
        config["training"]["save_frequency"] = 2
        train_metrics = EpochMetrics(1.0, 0.5, 0.4, 2)
        validation = EvaluationResult(
            EpochMetrics(1.0, 0.5, 0.4, 2),
            Predictions(["a"], [0], [0], [[1.0, 0.0]]),
        )
        saved_paths: list[tuple[Path, bool, int]] = []

        def record_checkpoint(path, payload, *, resume=False):
            saved_paths.append((path, resume, payload["epoch"]))

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                patch("dlbench.a1.trainer.train_one_epoch", return_value=train_metrics), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=validation), \
                patch("dlbench.a1.trainer.append_history"), \
                patch("dlbench.a1.trainer.save_metrics"), \
                patch("dlbench.a1.trainer.save_checkpoint", side_effect=record_checkpoint), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config)

        last_epochs = [epoch for path, resume, epoch in saved_paths if resume]
        self.assertEqual(last_epochs, [1, 3, 4])

    def test_fit_supports_rmsprop_optimizer(self) -> None:
        config = self._config("runs")
        config["training"]["optimizer"] = "rmsprop"
        captured_optimizers: list[torch.optim.Optimizer] = []
        train_metrics = EpochMetrics(1.0, 0.5, 0.4, 2)
        validation = EvaluationResult(
            EpochMetrics(0.9, 0.5, 0.4, 2),
            Predictions(["a"], [0], [0], [[1.0, 0.0]]),
        )

        def capture_optimizer(model, loader, optimizer, criterion, device):
            captured_optimizers.append(optimizer)
            return train_metrics

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                patch("dlbench.a1.trainer.train_one_epoch", side_effect=capture_optimizer), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=validation), \
                patch("dlbench.a1.trainer.append_history"), \
                patch("dlbench.a1.trainer.save_metrics"), \
                patch("dlbench.a1.trainer.save_checkpoint"), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split", "statistics_hash": "stats", "git_revision": "git"
                }):
            fit(config)

        self.assertEqual(len(captured_optimizers), 2)
        optimizer = captured_optimizers[0]
        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.defaults["lr"], 0.1)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)
        self.assertEqual(optimizer.defaults["momentum"], 0.8)

    def test_fit_rejects_unknown_optimizer(self) -> None:
        config = self._config("runs")
        config["training"]["optimizer"] = "madeup"

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.seed_everything"), \
                patch("dlbench.a1.trainer.create_run_dir", return_value=Path("runs/test")), \
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)):
            with self.assertRaisesRegex(
                ValueError,
                "Unsupported optimizer: madeup. Supported optimizers are adam, adamw, sgd, rmsprop, adagrad, adadelta, adamax, nadam.",
            ):
                fit(config)


class TestEvaluateCheckpoint(unittest.TestCase):
    @staticmethod
    def _config() -> dict[str, Any]:
        return {
            "protocol": {"id": "a1-v0"},
            "data": {"split_file": "split.json"},
            "preprocessing": {"mean": [0.1], "std": [0.2]},
            "evaluation": {"labels": list(range(10))},
            "checkpoint": {"monitor": "val_macro_f1"},
            "model": {"name": "linear", "parameters": {"input_dim": 1, "num_classes": 2}},
        }

    def test_invalid_split_is_rejected_before_loading(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_checkpoint({}, Path("missing.pt"), split=cast(Any, "train"))

    def test_evaluate_checkpoint_uses_validation_loader_and_model_weights(self) -> None:
        config = self._config()
        model = torch.nn.Linear(1, 2)
        loaders = SimpleNamespace(validation=object(), test=object())
        expected = EvaluationResult(
            EpochMetrics(0.25, 0.9, 0.85, 4),
            Predictions(["v1"], [1], [1], [[0.1, 0.9]]),
        )
        payload = {
            "config": config,
            "model_state_dict": model.state_dict(),
            "split_hash": "split-hash",
            "statistics_hash": "stats-hash",
        }

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.get_device", return_value=torch.device("cpu")), \
                patch("dlbench.a1.trainer.load_checkpoint", return_value=payload), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split-hash", "statistics_hash": "stats-hash", "git_revision": "git"
                }), \
                patch("dlbench.a1.trainer.build_model", return_value=model), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=loaders), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=expected) as evaluate:
            result = evaluate_checkpoint(config, Path("checkpoint.pt"), split="validation")

        self.assertIs(result, expected)
        self.assertIs(evaluate.call_args.args[1], loaders.validation)

    def test_evaluate_checkpoint_uses_test_loader_when_requested(self) -> None:
        config = self._config()
        model = torch.nn.Linear(1, 2)
        loaders = SimpleNamespace(validation=object(), test=object())
        expected = EvaluationResult(
            EpochMetrics(0.2, 0.95, 0.9, 4),
            Predictions(["t1"], [1], [1], [[0.05, 0.95]]),
        )
        payload = {
            "config": config,
            "model_state_dict": model.state_dict(),
            "split_hash": "split-hash",
            "statistics_hash": "stats-hash",
        }

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.get_device", return_value=torch.device("cpu")), \
                patch("dlbench.a1.trainer.load_checkpoint", return_value=payload), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "split-hash", "statistics_hash": "stats-hash", "git_revision": "git"
                }), \
                patch("dlbench.a1.trainer.build_model", return_value=model), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=loaders), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=expected) as evaluate:
            result = evaluate_checkpoint(config, Path("checkpoint.pt"), split="test")

        self.assertIs(result, expected)
        self.assertIs(evaluate.call_args.args[1], loaders.test)

    def test_evaluate_checkpoint_rejects_split_hash_mismatch(self) -> None:
        config = self._config()
        payload = {
            "config": config,
            "model_state_dict": torch.nn.Linear(1, 2).state_dict(),
            "split_hash": "old-split",
            "statistics_hash": "stats-hash",
        }

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.get_device", return_value=torch.device("cpu")), \
                patch("dlbench.a1.trainer.load_checkpoint", return_value=payload), \
                patch("dlbench.a1.trainer._checkpoint_metadata", return_value={
                    "split_hash": "current-split", "statistics_hash": "stats-hash", "git_revision": "git"
                }), \
                patch("dlbench.a1.trainer.build_model") as build_model:
            with self.assertRaisesRegex(ValueError, "split"):
                evaluate_checkpoint(config, Path("checkpoint.pt"))

        build_model.assert_not_called()

    def test_evaluate_checkpoint_rejects_model_config_mismatch(self) -> None:
        config = self._config()
        checkpoint_config = dict(config)
        checkpoint_config["model"] = {
            "name": "linear",
            "parameters": {"input_dim": 2, "num_classes": 2},
        }
        payload = {
            "config": checkpoint_config,
            "model_state_dict": torch.nn.Linear(1, 2).state_dict(),
            "split_hash": "split-hash",
            "statistics_hash": "stats-hash",
        }

        with patch("dlbench.a1.trainer.validate_config"), \
                patch("dlbench.a1.trainer.get_device", return_value=torch.device("cpu")), \
                patch("dlbench.a1.trainer.load_checkpoint", return_value=payload):
            with self.assertRaisesRegex(ValueError, "model"):
                evaluate_checkpoint(config, Path("checkpoint.pt"))


if __name__ == "__main__":
    unittest.main()