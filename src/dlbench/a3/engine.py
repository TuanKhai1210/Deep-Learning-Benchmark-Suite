"""Shared cached-feature training and held-out test evaluation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from dlbench.common.artifacts import create_run_dir
from dlbench.common.reproducibility import collect_environment, seed_everything

from .data import CLASSES, load_manifest, manifest_digest, publish_file
from .features import WIDTHS, load_cache
from .model import MODALITIES, ProductClassifier

LABEL_IDS = {name: index for index, name in enumerate(CLASSES)}


def _write_json(path, value):
    publish_file(path, lambda staged: staged.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    ))


def _run_git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), *args],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prepare(manifest, cache_paths, variant):
    rows = sorted(load_manifest(manifest), key=lambda row: row["sample_id"])
    features, cache_info = {}, {}
    for modality in MODALITIES[variant]:
        path = cache_paths.get(modality)
        if path is None:
            raise ValueError(f"{variant} requires --{modality}-cache")
        features[modality], cache_info[modality] = load_cache(path, rows, modality)
    return rows, features, cache_info


def _partition(rows, split):
    return [index for index, row in enumerate(rows) if row["split"] == split]


def _batch(indices, features, targets, device):
    return {name: values[indices].to(device) for name, values in features.items()}, targets[indices].to(device)


def classification_metrics(target, prediction):
    """Aggregate the entire split before computing macro-F1 on the fixed label set."""
    num_classes = len(CLASSES)
    matrix = torch.bincount(num_classes * target + prediction,
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    correct = matrix.diag()
    support, predicted = matrix.sum(1), matrix.sum(0)
    f1 = 2 * correct.double() / (support + predicted).clamp_min(1)
    return {
        "accuracy": float(correct.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {"label": label, "support": int(support[i]), "f1": float(f1[i])}
            for i, label in enumerate(CLASSES)
        ],
    }


@torch.inference_mode()
def _score(model, indices, features, targets, batch_size, device):
    model.eval()
    prediction = []
    loss_total = 0.0
    for batch in DataLoader(indices, batch_size=batch_size):
        values, labels = _batch(batch, features, targets, device)
        logits = model(values)
        loss_total += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        prediction.append(logits.argmax(1).cpu())
    pred = torch.cat(prediction)
    metrics = classification_metrics(targets[indices], pred)
    metrics["loss"] = loss_total / len(indices)
    if not math.isfinite(metrics["loss"]):
        raise ValueError("Non-finite evaluation loss")
    return metrics, pred


def _seed(seed, device):
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed)


def train(manifest, cache_paths, config, output_root, run_id, device):
    variant = config["model"]
    rows, features, cache_info = _prepare(manifest, cache_paths, variant)
    _seed(config["seed"], device)
    targets = torch.tensor([LABEL_IDS[row["label"]] for row in rows])
    train_loader = DataLoader(
        _partition(rows, "train"), batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    validation = _partition(rows, "validation")
    model = ProductClassifier(variant, WIDTHS, len(CLASSES), config["projection"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    run_dir = create_run_dir(Path(output_root), run_id)
    identity = {
        "schema": 1, "config": config, "manifest_sha256": manifest_digest(rows),
        "caches": cache_info, "classes": list(CLASSES),
        "git_revision": _run_git("rev-parse", "HEAD"),
        "git_status": _run_git("status", "--porcelain"), "device": str(device),
        "environment": collect_environment(),
        "trainable_head_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    _write_json(run_dir / "run.json", identity)
    history = []
    best = None
    for epoch in range(1, config["epochs"] + 1):
        started = perf_counter()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            values, labels = _batch(batch, features, targets, device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(values), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        metrics, _ = _score(model, validation, features, targets, config["batch_size"], device)
        history.append({
            "epoch": epoch, "train_loss": train_loss / len(train_loader.dataset),
            "validation_loss": metrics["loss"], "validation_macro_f1": metrics["macro_f1"],
            "validation_accuracy": metrics["accuracy"], "epoch_seconds": perf_counter() - started,
        })
        if best is None or metrics["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, **metrics}
            publish_file(run_dir / "best.pt", lambda staged: torch.save({
                "identity": identity, "epoch": epoch, "state_dict": model.state_dict(),
            }, staged))
        _write_json(run_dir / "history.json", history)
    _write_json(run_dir / "validation.json", best)
    return run_dir


def evaluate_test(manifest, cache_paths, run_dir, device):
    run_dir = Path(run_dir)
    identity = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    selected = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    config = identity["config"]
    variant = config["model"]
    rows, features, cache_info = _prepare(manifest, cache_paths, variant)
    if (manifest_digest(rows) != identity["manifest_sha256"] or cache_info != identity["caches"]
            or list(CLASSES) != identity["classes"]):
        raise ValueError("Test inputs differ from the training run")
    checkpoint_bytes = (run_dir / "best.pt").read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if checkpoint["identity"] != identity or checkpoint["epoch"] != selected["epoch"]:
        raise ValueError("Checkpoint does not match the selected run")
    model = ProductClassifier(variant, WIDTHS, len(CLASSES), config["projection"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    indices = _partition(rows, "test")
    targets = torch.tensor([LABEL_IDS[row["label"]] for row in rows])
    metrics, predictions = _score(model, indices, features, targets, config["batch_size"], device)
    metrics.update({
        "best_epoch": checkpoint["epoch"], "split": "test", "device": str(device),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
    })
    test_dir = run_dir / "test"
    with TemporaryDirectory(dir=run_dir) as staging:
        result = Path(staging) / "test"
        result.mkdir()
        _write_predictions(result / "predictions.csv", rows, indices, predictions)
        _write_json(result / "metrics.json", metrics)
        result.rename(test_dir)
    return test_dir


def _write_predictions(path, rows, indices, predictions):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_id", "label", "prediction"))
        writer.writerows(
            (rows[index]["sample_id"], rows[index]["label"], CLASSES[pred])
            for index, pred in zip(indices, predictions.tolist())
        )
