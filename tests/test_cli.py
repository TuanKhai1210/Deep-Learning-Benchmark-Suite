"""CLI orchestration tests; no ML imports, dataset downloads or training."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from dlbench.a1.cli import main
from dlbench.a1.contracts import EpochMetrics, EvaluationResult, FitResult, Predictions
from dlbench.common.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a1/models/linear.py"


class CLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        # Exercise draft gates independently of the checked-in experiment status.
        self.config["protocol"].update(status="draft", approved_by=[])
        self.original = deepcopy(self.config)
        self.calls = []
        self.result = EvaluationResult(
            EpochMetrics(0.4, 0.8, 0.75, 10), Predictions([], [], [], [])
        )
        self.trainer = ModuleType("dlbench.a1.trainer")

        def fit(config, *, smoke=False, resume_from=None):
            self.calls.append(("fit", config, smoke, resume_from))
            return FitResult(Path("runs/example"), Path("best.pt"), Path("history.csv"))

        def evaluate(config, checkpoint_path, *, split="validation", smoke=False):
            self.calls.append(("evaluate", config, checkpoint_path, split, smoke))
            return self.result

        self.trainer.fit = fit
        self.trainer.evaluate_checkpoint = evaluate
        self.dataset = ModuleType("dlbench.a1.data.dataset")
        self.dataset.prepare_data = Mock(return_value={"split_seed": 36})
        self.analysis = ModuleType("dlbench.a1.analysis")
        self.analysis.analyze_run = Mock()
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {
            "dlbench.a1.trainer": self.trainer,
            "dlbench.a1.data.dataset": self.dataset,
            "dlbench.a1.analysis": self.analysis,
        }).start()
        patch("dlbench.a1.cli.load_config", side_effect=lambda _: deepcopy(self.config)).start()

    def invoke(self, command, *options):
        args = [command]
        if command != "analyze":
            args += ["--config", str(CONFIG)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args + list(options)), 0)
        return output.getvalue()

    def reject(self, command, *options, message):
        with redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as caught:
                self.invoke(command, *options)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, output.getvalue())
        self.assertEqual(self.calls, [])

    def freeze(self):
        # Synthetic values exercise validation, not measured experiment settings.
        self.config["protocol"].update(status="frozen", approved_by=["A", "B", "C"])
        self.config["preprocessing"].update(mean=[0.5], std=[0.25])
        self.config["budget"].update(max_epochs=2, tuning_trials_per_model=1)
        self.config["training"]["batch_size"] = 2
        self.config["timing"].update(batch_size=2, warmup_steps=1, measurement_steps=1, device="cpu")

    def test_prepare_dispatches_and_prints_metadata(self):
        self.assertEqual(json.loads(self.invoke("prepare")), {"split_seed": 36})
        self.dataset.prepare_data.assert_called_once_with(self.config)

    def test_main_schema_and_split_path_are_preserved(self):
        self.assertIn("augmentations", self.config["preprocessing"])
        self.assertNotIn("augmentation", self.config["preprocessing"])
        self.assertIs(type(self.config["data"]["download"]), bool)
        self.assertEqual(Path(self.config["data"]["split_file"]).name,
                         "fashion_mnist_split.json")
        self.assertTrue(Path(self.config["data"]["split_file"]).is_file())
        self.invoke("train", "--smoke")
        self.assertEqual(self.calls[0][1]["preprocessing"], self.original["preprocessing"])

    def test_smoke_train_passes_seed_without_changing_split_or_source(self):
        self.assertIn("runs", self.invoke("train", "--smoke", "--seed", "69"))
        _, config, smoke, resume = self.calls[0]
        self.assertEqual(config["run"]["seed"], 69)
        self.assertEqual(config["data"], self.original["data"])
        self.assertTrue(smoke)
        self.assertIsNone(resume)
        self.assertEqual(self.config, self.original)

    def test_main_train_requires_frozen_config(self):
        self.reject("train", message="Not ready")

    def test_frozen_main_train_dispatches(self):
        self.freeze()
        self.invoke("train")
        self.assertFalse(self.calls[0][2])

    def test_resume_passes_checkpoint_path(self):
        self.invoke("train", "--smoke", "--resume-from", "last.pt")
        self.assertEqual(self.calls[0][3], Path("last.pt"))

    def test_resume_rejects_old_trainer_before_calling_it(self):
        def old_fit(config, *, smoke=False):
            self.fail("Unsupported trainer must not be called")
        self.trainer.fit = old_fit
        self.reject("train", "--smoke", "--resume-from", "last.pt", message="does not support resume_from")

    def test_smoke_evaluate_passes_arguments_and_prints_json(self):
        output = self.invoke("evaluate", "--smoke", "--checkpoint", "best.pt")
        self.assertEqual(self.calls[0][2:], (Path("best.pt"), "validation", True))
        self.assertEqual(json.loads(output)["macro_f1"], 0.75)

    def test_smoke_evaluate_rejects_old_trainer(self):
        def old_evaluate(config, checkpoint_path, *, split="validation"):
            self.fail("Unsupported evaluator must not be called")
        self.trainer.evaluate_checkpoint = old_evaluate
        self.reject("evaluate", "--smoke", "--checkpoint", "best.pt", message="does not support smoke")

    def test_smoke_test_is_rejected_even_with_permission(self):
        self.reject("evaluate", "--smoke", "--checkpoint", "best.pt", "--split", "test", "--allow-test", message="Smoke evaluation cannot")

    def test_test_requires_explicit_permission(self):
        self.reject("evaluate", "--checkpoint", "best.pt", "--split", "test", message="requires --allow-test")

    def test_test_permission_does_not_bypass_draft_validation(self):
        self.reject("evaluate", "--checkpoint", "best.pt", "--split", "test", "--allow-test", message="Not ready")

    def test_main_validation_rejects_draft(self):
        self.reject("evaluate", "--checkpoint", "best.pt", message="Not ready")

    def test_frozen_test_evaluation_dispatches(self):
        self.freeze()
        self.invoke("evaluate", "--checkpoint", "best.pt", "--split", "test", "--allow-test")
        self.assertEqual(self.calls[0][3], "test")

    def test_analyze_dispatches_without_loading_config(self):
        with patch("dlbench.a1.cli.load_config") as loader:
            self.invoke("analyze", "--run-dir", "runs/example")
        loader.assert_not_called()
        self.analysis.analyze_run.assert_called_once_with(Path("runs/example"))

    def test_backend_file_error_has_clear_exit(self):
        self.dataset.prepare_data.side_effect = FileNotFoundError("Split manifest missing")
        self.reject("prepare", message="Split manifest missing")


if __name__ == "__main__":
    unittest.main()
