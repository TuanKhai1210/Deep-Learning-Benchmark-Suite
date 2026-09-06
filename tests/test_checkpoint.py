"""Unit tests cho module dlbench.a1.checkpoint."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from dlbench.a1.checkpoint import is_better, load_checkpoint, save_checkpoint
from dlbench.a1.contracts import EpochMetrics


class TestCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_is_better_none_incumbent(self) -> None:
        """Epoch đầu tiên (incumbent is None) luôn được chấp nhận."""
        candidate = {"macro_f1": 0.82, "loss": 0.50, "epoch": 3}
        self.assertTrue(is_better(candidate, None))

    def test_is_better_macro_f1_priority(self) -> None:
        """Ưu tiên cao nhất là macro_f1 lớn hơn (kể cả khi loss cao hơn)."""
        incumbent = {"val_macro_f1": 0.80, "val_loss": 0.40, "epoch": 2}
        candidate = {"val_macro_f1": 0.82, "val_loss": 0.50, "epoch": 3}
        self.assertTrue(is_better(candidate, incumbent))

        worse_f1 = {"val_macro_f1": 0.79, "val_loss": 0.30, "epoch": 3}
        self.assertFalse(is_better(worse_f1, incumbent))

    def test_is_better_loss_tie_breaking(self) -> None:
        """Nếu macro_f1 hòa, ưu tiên loss thấp hơn."""
        incumbent = {"val_macro_f1": 0.85, "val_loss": 0.35, "epoch": 2}
        candidate = {"val_macro_f1": 0.85, "val_loss": 0.31, "epoch": 3}
        self.assertTrue(is_better(candidate, incumbent))

        worse_loss = {"macro_f1": 0.85, "loss": 0.39, "epoch": 3}
        self.assertFalse(is_better(worse_loss, incumbent))

    def test_is_better_epoch_tie_breaking(self) -> None:
        """Nếu cả F1 và loss đều hòa, ưu tiên epoch diễn ra sớm hơn."""
        incumbent = {"val_macro_f1": 0.85, "val_loss": 0.35, "epoch": 4}
        candidate = {"val_macro_f1": 0.85, "val_loss": 0.35, "epoch": 2}
        self.assertTrue(is_better(candidate, incumbent))

        later_epoch = {"macro_f1": 0.85, "loss": 0.35, "epoch": 6}
        self.assertFalse(is_better(later_epoch, incumbent))

    def test_is_better_rejects_nonfinite(self) -> None:
        """Từ chối candidate nếu F1 hoặc loss là NaN/Inf."""
        incumbent = {"val_macro_f1": 0.50, "val_loss": 1.0, "epoch": 1}
        nan_f1 = {"val_macro_f1": float("nan"), "val_loss": 0.5, "epoch": 2}
        inf_loss = {"val_macro_f1": 0.90, "val_loss": float("inf"), "epoch": 2}

        self.assertFalse(is_better(nan_f1, incumbent))
        self.assertFalse(is_better(inf_loss, incumbent))

    def test_save_and_load_checkpoint_roundtrip(self) -> None:
        """Lưu checkpoint đầy đủ schema và đọc lại chính xác weights/metadata."""
        ckpt_path = self.temp_dir / "checkpoints" / "best.pt"
        payload = {
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0, 3.0])},
            "epoch": 3,
            "config": {"model": "linear"},
            "val_metrics": {"accuracy": 0.85, "macro_f1": 0.84},
            "run_seed": 69420,
        }

        save_checkpoint(ckpt_path, payload)
        self.assertTrue(ckpt_path.is_file())

        loaded_payload = load_checkpoint(ckpt_path, map_location="cpu")
        self.assertEqual(loaded_payload["epoch"], 3)
        self.assertEqual(loaded_payload["run_seed"], 69420)
        self.assertTrue(
            torch.equal(
                loaded_payload["model_state_dict"]["weight"],
                payload["model_state_dict"]["weight"],
            )
        )

    def test_save_checkpoint_missing_keys_raises(self) -> None:
        """Báo lỗi khi payload thiếu key bắt buộc."""
        ckpt_path = self.temp_dir / "incomplete.pt"
        bad_payload = {
            "model_state_dict": {},
            "epoch": 1,
            # Thiếu config, val_metrics, run_seed
        }
        with self.assertRaises(ValueError):
            save_checkpoint(ckpt_path, bad_payload)

    def test_load_nonexistent_checkpoint_raises(self) -> None:
        """Báo lỗi FileNotFoundError nếu checkpoint không tồn tại."""
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.temp_dir / "nonexistent.pt")

    def test_resume_model_and_optimizer_state(self) -> None:
        """Verify model weights and optimizer internal momentum restore correctly."""
        # 1. Initialize source model and optimizer with momentum
        model_src = torch.nn.Linear(5, 2)
        optimizer_src = torch.optim.SGD(model_src.parameters(), lr=0.01, momentum=0.9)

        # Run one step of forward + backward + step to populate momentum buffers
        dummy_input = torch.randn(4, 5)
        loss = model_src(dummy_input).sum()
        loss.backward()
        optimizer_src.step()

        # 2. Package checkpoint payload representing last.pt at epoch 3
        ckpt_path = self.temp_dir / "last.pt"
        payload = {
            "model_state_dict": model_src.state_dict(),
            "optimizer_state_dict": optimizer_src.state_dict(),
            "epoch": 3,
            "config": {"model": "linear"},
            "val_metrics": {"accuracy": 0.80, "macro_f1": 0.79},
            "run_seed": 69420,
        }
        save_checkpoint(ckpt_path, payload)

        # 3. Create fresh destination model and optimizer
        model_dst = torch.nn.Linear(5, 2)
        optimizer_dst = torch.optim.SGD(model_dst.parameters(), lr=0.01, momentum=0.9)

        # Confirm weights differ before restoring
        self.assertFalse(torch.equal(model_src.weight, model_dst.weight))

        # 4. Resume states from checkpoint
        loaded_payload = load_checkpoint(ckpt_path, map_location="cpu")
        model_dst.load_state_dict(loaded_payload["model_state_dict"])
        optimizer_dst.load_state_dict(loaded_payload["optimizer_state_dict"])
        next_epoch = loaded_payload["epoch"] + 1

        # 5. Assert equality of resumed components
        self.assertEqual(next_epoch, 4)
        self.assertTrue(torch.equal(model_src.weight, model_dst.weight))
        self.assertTrue(torch.equal(model_src.bias, model_dst.bias))

        # Verify optimizer momentum buffers match exactly
        src_state = optimizer_src.state_dict()["state"]
        dst_state = optimizer_dst.state_dict()["state"]
        self.assertEqual(len(src_state), len(dst_state))
        for p_idx in src_state:
            self.assertTrue(
                torch.equal(
                    src_state[p_idx]["momentum_buffer"],
                    dst_state[p_idx]["momentum_buffer"],
                )
            )

    def test_resume_rng_state_reproducibility(self) -> None:
        """Verify RNG state restoration reproduces the subsequent random sequence."""
        # 1. Seed and generate an initial random stream
        torch.manual_seed(69420)
        _ = torch.rand(10)  # Consume initial random values

        # 2. Persist current RNG state in checkpoint
        ckpt_path = self.temp_dir / "rng_ckpt.pt"
        saved_rng = torch.get_rng_state()
        payload = {
            "model_state_dict": {},
            "epoch": 2,
            "config": {"model": "linear"},
            "val_metrics": {"accuracy": 0.8},
            "run_seed": 69420,
            "rng_state": {"torch_cpu": saved_rng},
        }
        save_checkpoint(ckpt_path, payload)

        # Generate reference values from the uninterrupted stream
        expected_next_numbers = torch.rand(5)

        # 3. Perturb global RNG state with a different seed
        torch.manual_seed(9999)
        _ = torch.rand(5)

        # 4. Load checkpoint and restore RNG state
        loaded_payload = load_checkpoint(ckpt_path, map_location="cpu")
        torch.set_rng_state(loaded_payload["rng_state"]["torch_cpu"])

        # Restored stream must match expected values identically
        restored_numbers = torch.rand(5)
        self.assertTrue(
            torch.equal(expected_next_numbers, restored_numbers),
            "Random numbers generated after resume do not match the expected sequence.",
        )

    def test_load_checkpoint_with_resume_false_ignores_missing_optimizer(self) -> None:
        """When resume=False, checkpoint loading succeeds without optimizer or scheduler states."""
        ckpt_path = self.temp_dir / "eval_ckpt.pt"
        minimal_payload = {
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "epoch": 2,
            "config": {"model": "mlp"},
            "val_metrics": {"accuracy": 0.82, "macro_f1": 0.81},
            "run_seed": 69420,
        }
        save_checkpoint(ckpt_path, minimal_payload)

        # Loading with resume=False must pass smoothly for evaluation/inference
        loaded = load_checkpoint(ckpt_path, map_location="cpu", resume=False)
        self.assertIn("model_state_dict", loaded)
        self.assertNotIn("optimizer_state_dict", loaded)
        self.assertEqual(loaded["epoch"], 2)

    def test_load_checkpoint_with_resume_true_validates_required_resume_states(self) -> None:
        """When resume=True, loading fails or raises if optimizer state is missing."""
        ckpt_path = self.temp_dir / "incomplete_for_resume.pt"
        payload_without_optimizer = {
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "epoch": 2,
            "config": {"model": "mlp"},
            "val_metrics": {"accuracy": 0.82, "macro_f1": 0.81},
            "run_seed": 69420,
            # optimizer_state_dict is intentionally omitted
        }
        save_checkpoint(ckpt_path, payload_without_optimizer)

        # If strict validation is enforced for resume mode, it should raise KeyError or ValueError
        with self.assertRaises((KeyError, ValueError)):
            load_checkpoint(ckpt_path, map_location="cpu", resume=True)

    def test_resume_model_optimizer_and_scheduler_state(self) -> None:
        """Verify model weights, optimizer momentum, scheduler, and RNG state restore correctly."""
        # 1. Initialize source model, optimizer with momentum, and learning rate scheduler
        model_src = torch.nn.Linear(5, 2)
        optimizer_src = torch.optim.SGD(model_src.parameters(), lr=0.01, momentum=0.9)
        scheduler_src = torch.optim.lr_scheduler.StepLR(optimizer_src, step_size=1, gamma=0.5)

        # Run one step to populate momentum buffers and advance scheduler
        dummy_input = torch.randn(4, 5)
        loss = model_src(dummy_input).sum()
        loss.backward()
        optimizer_src.step()
        scheduler_src.step()

        # 2. Package checkpoint payload representing last.pt at epoch 3
        ckpt_path = self.temp_dir / "last.pt"
        payload = {
            "model_state_dict": model_src.state_dict(),
            "optimizer_state_dict": optimizer_src.state_dict(),
            "scheduler_state_dict": scheduler_src.state_dict(),
            "epoch": 3,
            "config": {"model": "linear"},
            "val_metrics": {"accuracy": 0.80, "macro_f1": 0.79},
            "run_seed": 69420,
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
            },
        }
        save_checkpoint(ckpt_path, payload)

        # 3. Create fresh destination model, optimizer, and scheduler
        model_dst = torch.nn.Linear(5, 2)
        optimizer_dst = torch.optim.SGD(model_dst.parameters(), lr=0.01, momentum=0.9)
        scheduler_dst = torch.optim.lr_scheduler.StepLR(optimizer_dst, step_size=1, gamma=0.5)

        # Confirm weights differ before restoring
        self.assertFalse(torch.equal(model_src.weight, model_dst.weight))

        # 4. Resume states from checkpoint
        loaded_payload = load_checkpoint(ckpt_path, map_location="cpu", resume=True)
        model_dst.load_state_dict(loaded_payload["model_state_dict"])
        optimizer_dst.load_state_dict(loaded_payload["optimizer_state_dict"])
        scheduler_dst.load_state_dict(loaded_payload["scheduler_state_dict"])
        next_epoch = loaded_payload["epoch"] + 1

        # 5. Assert equality of resumed components
        self.assertEqual(next_epoch, 4)
        self.assertTrue(torch.equal(model_src.weight, model_dst.weight))
        self.assertTrue(torch.equal(model_src.bias, model_dst.bias))
        self.assertIn("rng_state", loaded_payload)

        # Verify optimizer momentum buffers match exactly
        src_opt_state = optimizer_src.state_dict()["state"]
        dst_opt_state = optimizer_dst.state_dict()["state"]
        self.assertEqual(len(src_opt_state), len(dst_opt_state))
        for p_idx in src_opt_state:
            self.assertTrue(
                torch.equal(
                    src_opt_state[p_idx]["momentum_buffer"],
                    dst_opt_state[p_idx]["momentum_buffer"],
                )
            )

        # Verify scheduler state (e.g. last_epoch and internal learning rates)
        self.assertEqual(scheduler_dst.last_epoch, scheduler_src.last_epoch)
        self.assertEqual(
            scheduler_dst.get_last_lr(),
            scheduler_src.get_last_lr(),
        )


if __name__ == "__main__":
    unittest.main()