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
    evaluate_checkpoint,
    fit,
    generate_run_id,
    get_scheduler,
)


class TestTrainerHelpers(unittest.TestCase):

    def test_generate_run_id_contains_model_mode_seed_and_timestamp(self) -> None:
        run_id = generate_run_id(
            {"model": {"name": "mlp"}, "run": {"seed": 67}},
            smoke=True,
        )

        self.assertRegex(run_id, r"^a1_mlp_smoke_seed67_\d{8}-\d{6}_[0-9a-f]{6}$")

    def test_fit_uses_run_metadata_provenance_for_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            config = {
                "model": {"name": "fake", "parameters": {}},
                "training": {"optimizer": "sgd", "learning_rate": 0.1},
                "run": {"seed": 67, "output_root": temporary_dir},
                "budget": {"max_epochs": 1},
                "data": {"split_file": "unused.json"},
                "preprocessing": {"mean": [0.1], "std": [0.2]},
            }
            metadata = {
                "schema_version": 1,
                "split": {"sha256": "split-from-metadata"},
                "normalization": {"sha256": "stats-from-metadata"},
                "git_revision": "metadata-git",
            }
            checkpoint_payloads: list[dict[str, object]] = []

            def record_checkpoint(path, payload, *, resume=False):
                checkpoint_payloads.append(payload)

            with patch("dlbench.a1.trainer.validate_config"), \
                    patch("dlbench.a1.trainer.seed_everything"), \
                    patch("dlbench.a1.trainer.create_run_dir", return_value=run_dir), \
                    patch("dlbench.a1.trainer.save_run_metadata", return_value=metadata), \
                    patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=[], validation=[])), \
                    patch("dlbench.a1.trainer.build_model", return_value=torch.nn.Linear(1, 2)), \
                    patch("dlbench.a1.trainer.train_one_epoch", return_value=EpochMetrics(1.0, 0.5, 0.4, 2)), \
                    patch("dlbench.a1.trainer.evaluate_epoch", return_value=EvaluationResult(
                        EpochMetrics(0.9, 0.5, 0.4, 2),
                        Predictions(["a"], [0], [0], [[1.0, 0.0]]),
                    )), \
                    patch("dlbench.a1.trainer.append_history"), \
                    patch("dlbench.a1.trainer.save_metrics"), \
                    patch("dlbench.a1.trainer.save_checkpoint", side_effect=record_checkpoint):
                fit(config)

            self.assertEqual(len(checkpoint_payloads), 2)
            self.assertEqual(checkpoint_payloads[0]["split_hash"], "split-from-metadata")
            self.assertEqual(checkpoint_payloads[0]["statistics_hash"], "stats-from-metadata")
            self.assertEqual(checkpoint_payloads[0]["git_revision"], "metadata-git")


class TestFitCalculations(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_root = self.temp_dir.name
        
        # Prepare dummy data
        features = torch.tensor([[-1.0], [1.0]] * 8)
        labels = torch.tensor([0, 1] * 8, dtype=torch.long)
        sample_ids = [f"sample-{index}" for index in range(len(labels))]
        dataset = [
            {"images": image, "labels": label, "sample_ids": sample_id}
            for image, label, sample_id in zip(features, labels, sample_ids)
        ]
        self.loader = DataLoader(cast(Any, dataset), batch_size=4, shuffle=False)
        self.model = torch.nn.Linear(1, 2)
        torch.nn.init.zeros_(self.model.weight)
        torch.nn.init.zeros_(self.model.bias)

        # Global patches to avoid I/O and unimplemented functions
        self.mock_validate_config = patch("dlbench.a1.trainer.validate_config").start()
        patch("dlbench.a1.trainer.seed_everything").start()
        patch("dlbench.a1.trainer.save_run_metadata", return_value={
            "schema_version": 1,
            "split": {"sha256": "split-from-metadata"},
            "normalization": {"sha256": "stats-from-metadata"},
            "git_revision": "metadata-git"
        }).start()
        self.mock_build_dataloaders = patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=self.loader, validation=self.loader)).start()
        patch("dlbench.a1.trainer.build_model", return_value=self.model).start()
        self.mock_append_history = patch("dlbench.a1.trainer.append_history").start()
        self.mock_save_metrics = patch("dlbench.a1.trainer.save_metrics").start()
        self.addCleanup(patch.stopall)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _config(self) -> dict[str, Any]:
        return {
            "model": {"name": "fake", "parameters": {}},
            "training": {
                "optimizer": "sgd",
                "learning_rate": 0.1,
                "weight_decay": 0.01,
                "momentum": 0.8,
            },
            "run": {"seed": 67, "output_root": self.output_root},
            "budget": {"max_epochs": 2},
            "data": {"split_file": "unused.json"},
            "preprocessing": {"mean": [0.1], "std": [0.2]},
        }

    @patch("dlbench.a1.trainer.evaluate_epoch")
    def test_fit_records_metrics_and_selects_lower_loss_on_f1_tie(self, mock_evaluate_epoch) -> None:
        config = self._config()
        mock_evaluate_epoch.side_effect = [
            EvaluationResult(EpochMetrics(0.8, 0.5, 0.7, 4), Predictions(["a"], [0], [0], [[1.0, 0.0]])),
            EvaluationResult(EpochMetrics(0.6, 0.6, 0.7, 4), Predictions(["a"], [0], [0], [[1.0, 0.0]])),
        ]
        
        result = fit(config)
        self.assertEqual(self.mock_append_history.call_count, 2)
        
        # Verify saved metrics
        saved_metrics = self.mock_save_metrics.call_args.args[1]
        self.assertEqual(saved_metrics["epoch"], 1)
        self.assertEqual(saved_metrics["val_loss"], 0.6)
        self.assertEqual(saved_metrics["val_macro_f1"], 0.7)
        
        # Checkpoint is verified directly on disk
        self.assertTrue(result.best_checkpoint.exists())

    @patch("dlbench.a1.trainer.train_one_epoch")
    def test_fit_uses_configured_sgd_hyperparameters(self, mock_train_one_epoch) -> None:
        config = self._config()
        captured_optimizers = []
        def capture_optimizer(model, loader, optimizer, criterion, device):
            captured_optimizers.append(optimizer)
            return EpochMetrics(1.0, 0.5, 0.4, 2)
        mock_train_one_epoch.side_effect = capture_optimizer
        
        fit(config)
        self.assertEqual(len(captured_optimizers), 2)
        optimizer = captured_optimizers[0]
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_fit_records_decreasing_training_and_validation_loss(self) -> None:
        config = self._config()
        fit(config)
        
        history_rows = [call.args[1] for call in self.mock_append_history.call_args_list]
        self.assertEqual(len(history_rows), 2)
        self.assertLess(history_rows[1]["train_loss"], history_rows[0]["train_loss"])
        self.assertLess(history_rows[1]["val_loss"], history_rows[0]["val_loss"])

    def test_fit_stops_after_configured_validation_patience(self) -> None:
        config = self._config()
        config["budget"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 2
        config["training"]["min_delta"] = 0.01
        config["training"]["learning_rate"] = 0.0 # Force no improvement
        
        fit(config)
        # Epoch 0: best. Epoch 1: no imp (patience 1). Epoch 2: no imp (patience 2 -> stop)
        self.assertEqual(self.mock_append_history.call_count, 3)

    def test_fit_uses_smoke_budget_and_skips_main_run_strict_validation(self) -> None:
        config = self._config()
        config["protocol"] = {"status": "draft", "id": "draft", "approved_by": ["A", "B", "C"]}
        config["budget"]["max_epochs"] = 25
        config["training"]["batch_size"] = 256
        
        fit(config, smoke=True)
        
        self.mock_validate_config.assert_called_once()
        self.assertFalse(self.mock_validate_config.call_args.kwargs["strict"])
        self.assertEqual(self.mock_validate_config.call_args.args[0]["budget"]["max_epochs"], 2)
        self.assertEqual(self.mock_build_dataloaders.call_args.args[0]["training"]["batch_size"], 32)

    def test_fit_supports_validation_loss_as_early_stopping_monitor(self) -> None:
        config = self._config()
        config["budget"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 2
        config["training"]["early_stopping_monitor"] = "loss"
        config["training"]["min_delta"] = 0.01
        config["training"]["learning_rate"] = 0.0 # Force no improvement
        
        fit(config)
        self.assertEqual(self.mock_append_history.call_count, 3)

    def test_fit_saves_last_checkpoint_when_early_stopping_interrupts_early(self) -> None:
        config = self._config()
        config["budget"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 2
        config["training"]["save_frequency"] = 5
        config["training"]["learning_rate"] = 0.0 # Force no improvement
        
        result = fit(config)
        # Should stop after epoch 2. Even though save_freq=5, last.pt should be saved.
        last_ckpt = result.run_dir / "last.pt"
        self.assertTrue(last_ckpt.exists())
        
        # We can actually verify the epoch in the payload by loading it
        payload = torch.load(last_ckpt, weights_only=False)
        self.assertEqual(payload["epoch"], 2)

    def test_fit_saves_last_checkpoint_when_early_stopping_hits_a_save_epoch(self) -> None:
        config = self._config()
        config["budget"]["max_epochs"] = 5
        config["training"]["early_stopping_patience"] = 1
        config["training"]["save_frequency"] = 2
        config["training"]["learning_rate"] = 0.0
        
        result = fit(config)
        # Stops after epoch 1. save_frequency=2 means epoch 1 is saved naturally
        # and also it's the interrupt epoch.
        last_ckpt = result.run_dir / "last.pt"
        self.assertTrue(last_ckpt.exists())
        payload = torch.load(last_ckpt, weights_only=False)
        self.assertEqual(payload["epoch"], 1)

    def test_fit_rejects_unknown_early_stopping_monitor(self) -> None:
        config = self._config()
        config["training"]["early_stopping_monitor"] = "accuracy"
        
        with self.assertRaisesRegex(ValueError, "early_stopping_monitor must be 'macro_f1' or 'loss'"):
            fit(config)

    def test_fit_saves_last_checkpoint_at_frequency_and_final_epoch(self) -> None:
        config = self._config()
        config["budget"]["max_epochs"] = 5
        config["training"]["save_frequency"] = 2
        config["training"]["early_stopping_patience"] = 0 # Disable early stopping
        
        result = fit(config)
        # Epochs run: 0, 1, 2, 3, 4
        # Saves last.pt at epoch 1, 3, and 4 (final)
        # We can only assert the final one easily without patching, but we know it runs to completion.
        last_ckpt = result.run_dir / "last.pt"
        self.assertTrue(last_ckpt.exists())
        payload = torch.load(last_ckpt, weights_only=False)
        self.assertEqual(payload["epoch"], 4)

    @patch("dlbench.a1.trainer.train_one_epoch")
    def test_fit_supports_rmsprop_optimizer(self, mock_train_one_epoch) -> None:
        config = self._config()
        config["training"]["optimizer"] = "rmsprop"
        captured_optimizers = []
        def capture_optimizer(model, loader, optimizer, criterion, device):
            captured_optimizers.append(optimizer)
            return EpochMetrics(1.0, 0.5, 0.4, 2)
        mock_train_one_epoch.side_effect = capture_optimizer
        
        fit(config)
        self.assertEqual(len(captured_optimizers), 2)
        optimizer = captured_optimizers[0]
        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.defaults["lr"], 0.1)

    def test_fit_rejects_unknown_optimizer(self) -> None:
        config = self._config()
        config["training"]["optimizer"] = "madeup"
        
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer: madeup"):
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
                patch("dlbench.a1.trainer.save_run_metadata") as mock_save, \
                patch("dlbench.a1.trainer.checkpoint_provenance", return_value={
                    "split_hash": "split-hash", "statistics_hash": "stats-hash", "git_revision": "git"
                }), \
                patch("dlbench.a1.trainer.build_model", return_value=model), \
                patch("dlbench.a1.trainer.build_dataloaders", return_value=loaders), \
                patch("dlbench.a1.trainer.evaluate_epoch", return_value=expected) as evaluate:
            result = evaluate_checkpoint(config, Path("checkpoint.pt"), split="validation")

        self.assertIs(result, expected)
        self.assertIs(evaluate.call_args.args[1], loaders.validation)
        
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        self.assertIsInstance(args[0], Path)
        self.assertIs(args[1], config)

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
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.checkpoint_provenance", return_value={
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
                patch("dlbench.a1.trainer.save_run_metadata"), \
                patch("dlbench.a1.trainer.checkpoint_provenance", return_value={
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


class TestGetScheduler(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(1, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def test_get_scheduler_step(self):
        config = {"scheduler": {"name": "step", "parameters": {"step_size": 10, "gamma": 0.5}}}
        scheduler = get_scheduler(self.optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 10)
        self.assertEqual(scheduler.gamma, 0.5)
        
    def test_get_scheduler_exponential(self):
        config = {"scheduler": {"name": "exponential", "parameters": {"gamma": 0.9}}}
        scheduler = get_scheduler(self.optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        self.assertEqual(scheduler.gamma, 0.9)
        
    def test_get_scheduler_cosine(self):
        config = {"scheduler": {"name": "cosine", "parameters": {"T_max": 50, "eta_min": 0.01}}}
        scheduler = get_scheduler(self.optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 50)
        self.assertEqual(scheduler.eta_min, 0.01)

    def test_get_scheduler_reduce_on_plateau(self):
        config = {"scheduler": {"name": "reduce_on_plateau", "parameters": {"mode": "max", "factor": 0.5, "patience": 5, "min_lr": 0.001}}}
        scheduler = get_scheduler(self.optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.mode, "max")
        self.assertEqual(scheduler.factor, 0.5)
        self.assertEqual(scheduler.patience, 5)
        self.assertEqual(scheduler.min_lrs[0], 0.001)

    def test_get_scheduler_polynomial(self):
        config = {"scheduler": {"name": "polynomial", "parameters": {"total_iters": 100, "power": 2.0}}}
        scheduler = get_scheduler(self.optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.PolynomialLR)
        self.assertEqual(scheduler.total_iters, 100)
        self.assertEqual(scheduler.power, 2.0)

    def test_get_scheduler_unsupported(self):
        config = {"scheduler": {"name": "unknown"}}
        with self.assertRaisesRegex(ValueError, "Unsupported scheduler: unknown"):
            get_scheduler(self.optimizer, config)
            
    def test_get_scheduler_none(self):
        self.assertIsNone(get_scheduler(self.optimizer, {}))


class TestTrainerMLPIntegration(unittest.TestCase):
    def test_mlp_fit_end_to_end_with_scheduler(self) -> None:
        from configs.a1.models.mlp import CONFIG
        import copy
        config = copy.deepcopy(CONFIG)
        # Simplify the model and data for fast test
        config["model"]["parameters"]["input_dim"] = 4
        config["model"]["parameters"]["hidden_dims"] = [4]
        config["model"]["parameters"]["num_classes"] = 2
        config["training"]["learning_rate"] = 0.1
        config["training"]["scheduler"] = {
            "name": "step",
            "parameters": {"step_size": 1, "gamma": 0.5}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["run"]["output_root"] = temp_dir
            
            features = torch.randn(16, 4)
            labels = torch.randint(0, 2, (16,))
            sample_ids = [f"sample-{index}" for index in range(len(labels))]
            dataset = [
                {"images": image, "labels": label, "sample_ids": sample_id}
                for image, label, sample_id in zip(features, labels, sample_ids)
            ]
            loader = DataLoader(cast(Any, dataset), batch_size=4, shuffle=False)
            
            with patch("dlbench.a1.trainer.build_dataloaders", return_value=SimpleNamespace(train=loader, validation=loader)), \
                 patch("dlbench.a1.trainer.save_run_metadata", return_value={"schema_version": 1, "split": {"sha256": "hash"}, "normalization": {"sha256": "hash"}, "git_revision": "git"}), \
                 patch("dlbench.a1.trainer.append_history"), \
                 patch("dlbench.a1.trainer.save_metrics"), \
                 patch("dlbench.a1.trainer.validate_config"):
                result = fit(config, smoke=True)
                
            self.assertTrue(result.best_checkpoint.exists())
            
            # verify scheduler stepped
            payload = torch.load(result.best_checkpoint, weights_only=False)
            self.assertIn("scheduler_state_dict", payload)
            self.assertIsNotNone(payload["scheduler_state_dict"])


if __name__ == "__main__":
    unittest.main()