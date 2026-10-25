"""Verified seed-36 manifest adapter and stateless paired augmentation.

Raw data stays external. Workers receive their epoch before iteration; persistent
workers are deliberately disabled so resumed epochs use the same augmentation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from .converter import (
    CONVERTER_VERSION, letterbox_target, load_records, rasterize_semantic,
)

MANIFEST_SHA256 = "87905f1a97dd2c04194e049ca61d1a7f3bbe2942e9fe17dc2d737c371cdce943"
SPLITS = ("train", "validation", "test")
CLASS_NAMES = (
    "background", "rov", "plant", "animal_fish", "animal_starfish",
    "animal_shells", "animal_crab", "animal_eel", "animal_etc", "trash_etc",
    "trash_fabric", "trash_fishing_gear", "trash_metal", "trash_paper",
    "trash_plastic", "trash_rubber", "trash_wood",
)


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_manifest(path, *, expected_digest=MANIFEST_SHA256):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = content_hash({k: v for k, v in manifest.items() if k != "content_sha256"})
    if digest != manifest.get("content_sha256") or digest != expected_digest:
        raise ValueError("Manifest content hash mismatch")
    if manifest["split_seed"] != 36 or manifest["rasterizer"]["version"] != CONVERTER_VERSION:
        raise ValueError("Unexpected split seed or converter")
    if manifest["categories"] != [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]:
        raise ValueError("Category map differs from TrashCan-Material")
    rows = manifest["images"]
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    if any(r["split"] not in SPLITS for r in rows):
        raise ValueError("Unknown partition")
    groups = [set(r["video_id"] for r in rows if r["split"] == s) for s in SPLITS]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Video leakage")
    if expected_digest == MANIFEST_SHA256 and Counter(r["split"] for r in rows) != {
        "train": 5057, "validation": 1077, "test": 1078,
    }:
        raise ValueError("Unexpected published split counts")
    return manifest


def _rng(seed, epoch, image_id, stream):
    # Neither sampler order nor model dropout consumes this RNG stream.
    token = f"a2-v1/{seed}/{epoch}/{image_id}/{stream}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], "big"))


def transform_pair(rgb, mask, *, size, mean, std, seed, epoch, image_id,
                   training, color_jitter):
    if training and color_jitter:
        rng = _rng(seed, epoch, image_id, "color")
        if rng.random() < 0.5:
            operations = [
                (TF.adjust_brightness, rng.uniform(0.9, 1.1)),
                (TF.adjust_contrast, rng.uniform(0.9, 1.1)),
                (TF.adjust_saturation, rng.uniform(0.9, 1.1)),
                (TF.adjust_hue, rng.uniform(-0.01, 0.01)),
            ]
            rng.shuffle(operations)
            for operation, factor in operations:
                rgb = operation(rgb, factor)
    rgb, mask = letterbox_target(rgb, mask, size=size)
    image = TF.normalize(TF.to_tensor(rgb), mean=mean, std=std)
    return image, torch.from_numpy(mask.copy()).long()


class TrashCanDataset(Dataset):
    def __init__(self, root, manifest_path, split, preprocessing, *, seed=69420,
                 color_jitter=False, expected_digest=MANIFEST_SHA256):
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")
        self.root = Path(root).resolve()
        manifest = read_manifest(manifest_path, expected_digest=expected_digest)
        if PIL.__version__ != manifest["rasterizer"]["pillow_version"]:
            raise ValueError("Pillow version differs from the frozen target converter")
        records, categories, hashes, _ = load_records(self.root)
        if hashes != manifest["source_annotation_sha256"]:
            raise ValueError("Source annotation hashes differ from manifest")
        if categories != {i: n for i, n in enumerate(CLASS_NAMES) if i}:
            raise ValueError("Source categories differ from manifest")
        by_id = {r["id"]: r for r in records}
        if set(by_id) != {r["image_id"] for r in manifest["images"]}:
            raise ValueError("Source image membership differs from manifest")
        self.rows = [r for r in manifest["images"] if r["split"] == split]
        for row in manifest["images"]:
            record = by_id[row["image_id"]]
            for field in ("relative_path", "source_partition", "file_name", "video_id"):
                if row[field] != record[field]:
                    raise ValueError(f"Manifest/source mismatch: {field}")
            path = (self.root / row["relative_path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Image path escapes dataset root")
        self.records, self.preprocessing = by_id, dict(preprocessing)
        self.split, self.seed, self.epoch = split, seed, 0
        self.color_jitter = bool(color_jitter)
        self.manifest_digest = manifest["content_sha256"]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        record = self.records[row["image_id"]]
        with Image.open(self.root / row["relative_path"]) as source:
            if source.size != (record["width"], record["height"]):
                raise ValueError("Image dimensions differ from annotations")
            rgb = source.convert("RGB")
        mask = rasterize_semantic(record)[0]
        if hashlib.sha256(mask.tobytes()).hexdigest() != row["target_sha256"]:
            raise ValueError(f"Native target hash mismatch: {row['image_id']}")
        image, target = transform_pair(
            rgb, mask, **self.preprocessing, seed=self.seed, epoch=self.epoch,
            image_id=row["image_id"], training=self.split == "train",
            color_jitter=self.color_jitter,
        )
        return {"image": image, "mask": target, "image_id": row["image_id"],
                "video_id": row["video_id"]}


def make_loader(dataset, *, batch_size, workers=0, epoch=0):
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(dataset.seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.split == "train",
                      num_workers=workers, generator=generator, persistent_workers=False,
                      drop_last=False)
