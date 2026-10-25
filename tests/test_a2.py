"""A2 initial-code acceptance: semantic contracts and epoch-boundary replay.

All data here are synthetic fixtures. This does not validate the external images,
main-run budget, Colab/CUDA or pretrained model quality.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import PIL
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset

from dlbench.a2.cli import main, synthetic_smoke
from dlbench.a2.config import load_config, validate_config
from dlbench.a2.converter import CONVERTER_VERSION, letterbox_target, rasterize_semantic
from dlbench.a2.data import (
    CLASS_NAMES, MANIFEST_SHA256, TrashCanDataset, content_hash, read_manifest, transform_pair,
)
from dlbench.a2.engine import run_epoch
from dlbench.a2.losses import SegmentationLoss
from dlbench.a2.metrics import SegmentationMetrics
from dlbench.a2.models import build_model, count_parameters
from dlbench.a2.models.unet import UNet
from dlbench.a2.trainer import fit, is_better, evaluate_checkpoint, state_hash
from dlbench.a2.comparison import color_decision

ROOT = Path(__file__).resolve().parents[1]


def config(name="unet"):
    return load_config(ROOT / f"configs/a2/{name}.py")


def polygon(category=1, identifier=1, coords=None):
    return {"id": identifier, "category_id": category,
            "segmentation": [coords or [1, 1, 5, 1, 5, 5, 1, 5]]}


class TinyDataset(Dataset):
    def __init__(self, split, cfg):
        self.split, self.seed, self.epoch = split, cfg["run"]["seed"], 0
        self.manifest_digest = MANIFEST_SHA256
        self.preprocessing = deepcopy(cfg["preprocessing"])
        self.color_jitter = cfg["run"]["color_jitter"]
        self.rows = [{"id": i, "split": split} for i in range(4)]
        generator = torch.Generator().manual_seed(123 if split == "train" else 456)
        self.images = torch.rand(4, 3, 8, 8, generator=generator)
        self.masks = torch.randint(0, 17, (4, 8, 8), generator=generator)
        self.masks[:, 0, :] = 255

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"image": self.images[index], "mask": self.masks[index],
                "image_id": index, "video_id": self.split}


def tiny_model(*args, **kwargs):
    return nn.Sequential(nn.Conv2d(3, 8, 1), nn.Dropout2d(0.1), nn.Conv2d(8, 17, 1))


class A2ParameterCountTests(unittest.TestCase):
    def test_total_trainable_frozen_exclude_buffers(self):
        model = nn.Linear(3, 2)
        model.register_buffer("not_a_parameter", torch.zeros(100))
        self.assertEqual(count_parameters(model), {
            "total_params": 8, "trainable_params": 8, "frozen_params": 0})
        model.weight.requires_grad_(False)
        self.assertEqual(count_parameters(model), {
            "total_params": 8, "trainable_params": 2, "frozen_params": 6})
        model.bias.requires_grad_(False)
        self.assertEqual(count_parameters(model), {
            "total_params": 8, "trainable_params": 0, "frozen_params": 8})

    def test_shared_parameters_count_once(self):
        layer = nn.Linear(3, 2)
        self.assertEqual(count_parameters(nn.Sequential(layer, layer))["total_params"], 8)

    def test_smoke_exposes_explicit_counts(self):
        result = synthetic_smoke(config(), size=32)
        self.assertEqual(result["total_params"], 1964369)
        self.assertEqual(result["trainable_params"], result["total_params"])
        self.assertEqual(result["frozen_params"], 0)
        self.assertNotIn("parameters", result)


class A2ConfigTests(unittest.TestCase):
    def test_three_configs_and_main_gate(self):
        for name in ("unet", "segformer_off", "segformer_on"):
            cfg = config(name)
            self.assertEqual(len(validate_config(cfg)), 3)
            with self.assertRaisesRegex(ValueError, "Main training blocked"):
                validate_config(cfg, main=True)

    def test_pair_only_color_and_name_differ(self):
        off, on = config("segformer_off"), config("segformer_on")
        for value in (off, on):
            value["run"].pop("name")
            value["run"].pop("color_jitter")
        self.assertEqual(off, on)

    def test_reject_invalid_optimizer_and_revision(self):
        cfg = config("segformer_off")
        cfg["model"]["revision"] = "main"
        with self.assertRaises(ValueError):
            validate_config(cfg)
        cfg = config()
        cfg["training"]["lr"] = float("nan")
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_literal_reader_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.py"
            path.write_text("import os\nCONFIG = {}\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_cli_validate(self):
        self.assertEqual(main(["validate-config", "--config", str(ROOT / "configs/a2/unet.py")]), 0)


class A2DataTests(unittest.TestCase):
    def test_converter_union_overlap_order_and_ignore(self):
        record = {"width": 8, "height": 8,
                  "annotations": [polygon(), polygon(identifier=2), polygon(2, 3, [3, 3, 7, 3, 7, 7, 3, 7])]}
        labels = rasterize_semantic(record)[0]
        self.assertEqual(labels[2, 2], 1)
        self.assertEqual(labels[4, 4], 255)
        self.assertEqual(labels[6, 6], 2)
        self.assertEqual(labels[0, 0], 0)
        record["annotations"].reverse()
        np.testing.assert_array_equal(labels, rasterize_semantic(record)[0])

    def test_converter_rejects_rle_and_degenerate(self):
        for segmentation in ({"counts": []}, [[0, 0, 1, 1, 2, 2]]):
            with self.assertRaises(ValueError):
                rasterize_semantic({"width": 4, "height": 4,
                                    "annotations": [{"id": 1, "category_id": 1,
                                                     "segmentation": segmentation}]})

    def test_letterbox_nearest_and_ignored_padding(self):
        rgb, mask = letterbox_target(Image.new("RGB", (8, 4)),
                                    np.ones((4, 8), dtype=np.uint8), size=16)
        self.assertEqual(rgb.size, (16, 16))
        self.assertTrue((mask[:4] == 255).all())
        self.assertTrue((mask[4:12] == 1).all())
        self.assertEqual(set(np.unique(mask)), {1, 255})

    def test_color_is_stateless_and_preserves_mask_eval(self):
        rgb = Image.fromarray(np.arange(8 * 4 * 3, dtype=np.uint8).reshape(4, 8, 3))
        mask = np.ones((4, 8), dtype=np.uint8)
        kwargs = dict(size=16, mean=[0, 0, 0], std=[1, 1, 1], seed=69420,
                      epoch=1, image_id=7, training=True)
        off = transform_pair(rgb, mask, color_jitter=False, **kwargs)
        before = torch.get_rng_state().clone()
        on = transform_pair(rgb, mask, color_jitter=True, **kwargs)
        repeated = transform_pair(rgb, mask, color_jitter=True, **kwargs)
        self.assertTrue(torch.equal(on[1], off[1]))
        self.assertTrue(torch.equal(on[0], repeated[0]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        kwargs["training"] = False
        self.assertTrue(torch.equal(transform_pair(rgb, mask, color_jitter=True, **kwargs)[0],
                                    transform_pair(rgb, mask, color_jitter=False, **kwargs)[0]))

    def test_fixture_dataset_native_hash_and_annotation_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            categories = [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES) if i]
            rows, hashes = [], {}
            for index, part in enumerate(("train", "val"), 1):
                (root / part).mkdir()
                name = f"vid_{index:06d}_frame0000001.jpg"
                Image.new("RGB", (8, 8), (50, 100, 150)).save(root / part / name)
                ann = {**polygon(), "id": index, "image_id": index}
                record = {"id": index, "width": 8, "height": 8, "file_name": name}
                source = root / f"instances_{part}_trashcan.json"
                source.write_text(json.dumps({"images": [record], "annotations": [ann],
                                              "categories": categories}), encoding="utf-8")
                hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
                rows.append({"image_id": index, "file_name": name,
                             "relative_path": f"{part}/{name}", "source_partition": part,
                             "video_id": f"vid_{index:06d}", "split": "train" if index == 1 else "validation",
                             "target_sha256": hashlib.sha256(rasterize_semantic(
                                 {**record, "annotations": [ann]})[0].tobytes()).hexdigest()})
            manifest = {"split_seed": 36, "rasterizer": {"version": CONVERTER_VERSION,
                         "pillow_version": PIL.__version__},
                        "source_annotation_sha256": hashes, "categories": [{"id": 0, "name": "background"}] + categories,
                        "images": rows}
            digest = content_hash(manifest)
            manifest["content_sha256"] = digest
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = TrashCanDataset(root, path, "train", dict(size=16, mean=[0]*3, std=[1]*3), expected_digest=digest)
            self.assertEqual(dataset[0]["image"].shape, (3, 16, 16))
            dataset.rows[0]["target_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "target hash"):
                dataset[0]
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises((ValueError, KeyError)):
                TrashCanDataset(root, path, "train", dict(size=16, mean=[0]*3, std=[1]*3), expected_digest=digest)

    def test_manifest_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"split_seed": 36, "content_sha256": "bad"}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_manifest(path)


class A2NumericsTests(unittest.TestCase):
    def test_metrics_background_false_positives_and_ignore(self):
        metrics = SegmentationMetrics()
        metrics.update(torch.tensor([1, 1, 0, 1]), torch.tensor([0, 1, 1, 255]))
        result = metrics.compute()
        self.assertAlmostEqual(result["foreground_miou"], 1 / 3)
        self.assertAlmostEqual(result["foreground_dice"], 0.5)
        self.assertEqual(result["ignore_fraction"], 0.25)
        self.assertIsNone(result["trash_miou"])
        self.assertEqual(result["per_class"][1]["support"], 2)

    def test_metrics_accumulate_partition_not_batch_macros(self):
        first = SegmentationMetrics()
        first.update(torch.tensor([1]), torch.tensor([1]))
        first.update(torch.tensor([0, 0, 0]), torch.tensor([1, 1, 1]))
        self.assertEqual(first.compute()["foreground_miou"], 0.25)

    def test_prediction_only_class_counts_zero_not_absent(self):
        metrics = SegmentationMetrics()
        metrics.update(torch.tensor([2]), torch.tensor([0]))
        self.assertEqual(metrics.compute()["foreground_miou"], 0)
        self.assertEqual(metrics.compute()["foreground_defined_classes"], 1)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(ValueError):
            SegmentationMetrics().update(torch.tensor([17]), torch.tensor([1]))

    def test_loss_ignore_has_zero_gradient_and_finite_empty(self):
        logits = torch.randn(1, 17, 2, 2, requires_grad=True)
        target = torch.tensor([[[1, 255], [0, 2]]])
        loss = SegmentationLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, :, 0, 1], torch.zeros(17)))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(SegmentationLoss()(logits, torch.full_like(target, 255)).detach()), 0)

    def test_ce_matches_reference_without_dice(self):
        logits = torch.randn(1, 17, 2, 2)
        target = torch.tensor([[[0, 1], [2, 255]]])
        self.assertAlmostEqual(float(SegmentationLoss(dice_weight=0)(logits, target)),
                               float(nn.functional.cross_entropy(logits, target, ignore_index=255)))

    def test_unet_shape_odd_dimensions_backward(self):
        model = UNet(base_channels=8)
        logits = model(torch.randn(2, 3, 33, 47))
        self.assertEqual(logits.shape, (2, 17, 33, 47))
        logits.mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_eval_restores_mode_and_does_not_update(self):
        model = tiny_model()
        model.train()
        before = state_hash(model)
        batch = TinyDataset("validation", config())[0]
        batch = {"image": batch["image"][None], "mask": batch["mask"][None]}
        run_epoch(model, [batch], SegmentationLoss(), "cpu")
        self.assertTrue(model.training)
        self.assertEqual(before, state_hash(model))

    def test_empty_supervision_does_not_apply_weight_decay(self):
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1)
        before = state_hash(model)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            run_epoch(model, [{"image": torch.zeros(1, 3, 8, 8),
                               "mask": torch.full((1, 8, 8), 255)}], SegmentationLoss(), "cpu", optimizer)
        self.assertEqual(before, state_hash(model))

    def test_checkpoint_ties(self):
        best = {"foreground_miou": 0.5, "loss": 1.0}
        self.assertFalse(is_better(dict(best), best))
        self.assertTrue(is_better({"foreground_miou": 0.5, "loss": 0.9}, best))
        with self.assertRaises(ValueError):
            is_better({"foreground_miou": None, "loss": 0}, best)


class A2CheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_optimizer_rng_and_history(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp, patch("dlbench.a2.trainer.build_model", tiny_model):
            train, val = TinyDataset("train", cfg), TinyDataset("validation", cfg)
            full, resumed = Path(tmp) / "full", Path(tmp) / "resumed"
            fit(cfg, train, val, full, development=True, epochs=2)
            metadata = json.loads((full / "metadata.json").read_text())
            self.assertEqual(metadata["parameter_counts"], count_parameters(tiny_model()))
            fit(cfg, train, val, resumed, development=True, epochs=1)
            (resumed / "history.json").write_text("interrupted")
            fit(cfg, train, val, resumed, development=True, epochs=2, resume=True)
            a = torch.load(full / "last.pt", weights_only=False)
            b = torch.load(resumed / "last.pt", weights_only=False)
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(a["best"], b["best"])
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            for key in a["optimizer"]["state"]:
                for field, value in a["optimizer"]["state"][key].items():
                    self.assertTrue(torch.equal(value, b["optimizer"]["state"][key][field]))
            self.assertEqual(json.loads((resumed / "history.json").read_text()), a["history"])
            replay = evaluate_checkpoint(resumed, val)
            self.assertEqual(replay["metrics"], a["history"][replay["selected_epoch"] - 1]["validation"])
            with self.assertRaisesRegex(ValueError, "development checkpoints"):
                evaluate_checkpoint(resumed, TinyDataset("test", cfg))
            with self.assertRaisesRegex(FileExistsError, ""):
                fit(cfg, train, val, resumed, development=True, epochs=1)
            cfg["training"]["lr"] *= 2
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                fit(cfg, train, val, resumed, development=True, epochs=2, resume=True)

    def test_train_rejects_test_partition_before_model(self):
        cfg = config()
        with self.assertRaisesRegex(ValueError, "train and validation only"):
            fit(cfg, TinyDataset("train", cfg), TinyDataset("test", cfg), "unused", development=True)


class A2ComparisonTests(unittest.TestCase):
    def pair(self):
        metadata, evaluations = [], []
        for name in ("segformer_off", "segformer_on"):
            identity = {"config": config(name), "source_digest": "abc", "device": "cpu",
                        "environment": {}, "dependencies": {}, "cpu_threads": 2,
                        "train_membership": "train", "validation_membership": "val", "development": False}
            digest = content_hash(identity)
            metadata.append({"identity": identity, "identity_sha256": digest, "initial_state_sha256": "same"})
            evaluations.append({"split": "validation", "kind": "main", "membership_sha256": "val", "run_identity_sha256": digest,
                                "metrics": {"trash_miou": 0.2, "foreground_miou": 0.3, "foreground_dice": 0.4}})
        return metadata, evaluations

    def test_threshold_and_no_test_selection(self):
        metadata, evaluations = self.pair()
        evaluations[1]["metrics"] = {"trash_miou": 0.21, "foreground_miou": 0.29, "foreground_dice": 0.39}
        self.assertTrue(color_decision(*metadata, *evaluations)["accept_color"])
        evaluations[1]["split"] = "test"
        with self.assertRaises(ValueError):
            color_decision(*metadata, *evaluations)

    def test_mismatched_initialization_rejected(self):
        metadata, evaluations = self.pair()
        metadata[1]["initial_state_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "initial_state"):
            color_decision(*metadata, *evaluations)

    def test_different_validation_membership_rejected(self):
        metadata, evaluations = self.pair()
        evaluations[1]["membership_sha256"] = "different-subset"
        with self.assertRaisesRegex(ValueError, "membership"):
            color_decision(*metadata, *evaluations)


class A2SegFormerTests(unittest.TestCase):
    def test_real_b0_scratch_forward_backward_and_paired_initialization(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("optional transformers dependency unavailable")
        from dlbench.common.reproducibility import seed_everything
        cfg = config("segformer_off")
        seed_everything(69420)
        first = build_model(cfg["model"], pretrained=False)
        digest = state_hash(first)
        seed_everything(69420)
        second = build_model(config("segformer_on")["model"], pretrained=False)
        self.assertEqual(digest, state_hash(second))
        logits = first(torch.randn(2, 3, 64, 64))
        self.assertEqual(logits.shape, (2, 17, 64, 64))
        SegmentationLoss()(logits, torch.ones(2, 64, 64, dtype=torch.long)).backward()
        self.assertTrue(first.network.decode_head.classifier.weight.grad is not None)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
