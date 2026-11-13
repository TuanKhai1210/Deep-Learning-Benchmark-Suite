"""Feature-cache interchange and frozen encoder adapters."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .data import CLASSES, manifest_digest, publish_file

WIDTHS = {"image": 512, "title": 768}
TEXT_MODEL = "distilbert/distilbert-base-uncased"
TEXT_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def mask_titles(titles, phrases=()):
    """Use one lexicon across all samples, without consulting their labels."""
    terms = {tuple(re.findall(r"[a-z0-9]+", value.lower()))
             for value in [*CLASSES, *phrases]}
    terms.discard(())
    options = sorted(terms, key=lambda term: (-len(term), term))
    pattern = re.compile(r"(?<![a-z0-9])(?:" + "|".join(
        r"[^a-z0-9]+".join(term) for term in options
    ) + r")(?![a-z0-9])", flags=re.IGNORECASE | re.ASCII)
    return [pattern.sub("[MASK]", title) for title in titles]


def _validate_features(ids, vectors, rows, modality):
    expected = {row["sample_id"] for row in rows}
    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise ValueError(f"{modality} cache IDs must be a Unicode vector")
    if len(ids) != len(expected) or set(ids.tolist()) != expected:
        raise ValueError(f"{modality} cache has duplicate, missing or extra IDs")
    if (vectors.shape != (len(ids), WIDTHS[modality]) or
            vectors.dtype != np.float32 or not np.isfinite(vectors).all()):
        raise ValueError(f"Invalid {modality} feature shape, dtype or values")


def save_cache(path, rows, modality, vectors, encoder):
    ids = np.asarray([row["sample_id"] for row in rows])
    _validate_features(ids, vectors, rows, modality)
    metadata = {
        "schema": 1,
        "modality": modality,
        "manifest_sha256": manifest_digest(rows),
        "encoder": encoder,
    }

    def write(staged):
        with staged.open("wb") as stream:
            np.savez(stream, ids=ids, features=vectors, metadata=json.dumps(metadata))

    publish_file(path, write)


def load_cache(path, rows, modality):
    payload = Path(path).read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        ids, vectors = archive["ids"], archive["features"]
        metadata = json.loads(archive["metadata"].item())
    if (not isinstance(metadata, dict) or metadata.get("schema") != 1 or metadata.get("modality") != modality
            or metadata.get("manifest_sha256") != manifest_digest(rows)
            or not isinstance(metadata.get("encoder"), dict)):
        raise ValueError(f"{modality} cache metadata does not match the manifest")
    _validate_features(ids, vectors, rows, modality)
    lookup = {sample: index for index, sample in enumerate(ids.tolist())}
    aligned = vectors[[lookup[row["sample_id"]] for row in rows]]
    return torch.from_numpy(aligned), {**metadata, "sha256": hashlib.sha256(payload).hexdigest()}


def _image_batch(rows, root, transform):
    from PIL import Image, ImageOps

    images = []
    for row in rows:
        path = (root / row["image_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image escapes image root: {row['sample_id']}")
        with Image.open(path) as image:
            images.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
    return torch.stack(images)


@torch.inference_mode()
def encode_images(rows, image_root, batch_size, device):
    from torchvision.models import ResNet18_Weights, resnet18

    root = Path(image_root).resolve(strict=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.requires_grad_(False).eval().to(device)
    transform = weights.transforms()
    started = perf_counter()
    vectors = []
    for start in range(0, len(rows), batch_size):
        images = _image_batch(rows[start:start + batch_size], root, transform)
        vectors.append(model(images.to(device)).cpu().numpy())
    return np.concatenate(vectors), {
        "model": "resnet18", "weights": weights.name, "weights_url": weights.url,
        "transform": "EXIF transpose, RGB, IMAGENET1K_V1.transforms",
        "device": str(device), "batch_size": batch_size,
        "seconds": perf_counter() - started,
    }


@torch.inference_mode()
def encode_titles(rows, batch_size, device, *, raw=False, phrases=()):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
    model = transformers.AutoModel.from_pretrained(
        TEXT_MODEL, revision=TEXT_REVISION, use_safetensors=True,
    ).requires_grad_(False).eval().to(device)
    titles = [row["title"] for row in rows]
    inputs = titles if raw else mask_titles(titles, phrases)
    started = perf_counter()
    vectors = []
    for start in range(0, len(rows), batch_size):
        tokens = tokenizer(inputs[start:start + batch_size], padding=True,
                           truncation=True, max_length=64, return_tensors="pt")
        vectors.append(model(**tokens.to(device)).last_hidden_state[:, 0].cpu().numpy())
    return np.concatenate(vectors), {
        "model": TEXT_MODEL, "revision": TEXT_REVISION, "max_length": 64,
        "title_policy": "raw" if raw else "masked",
        "additional_phrases": list(phrases),
        "titles_changed": sum(a != b for a, b in zip(titles, inputs)),
        "transformers": transformers.__version__, "device": str(device),
        "batch_size": batch_size, "seconds": perf_counter() - started,
    }
