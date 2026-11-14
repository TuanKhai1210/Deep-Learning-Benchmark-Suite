"""Run and compare Khải's three fusion variants on one locked cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from .data import load_manifest, manifest_digest, read_csv, write_csv
from .engine import LABEL_IDS, _write_json, classification_metrics, evaluate_test, train
from .features import load_cache

VARIANTS = ("concat", "product", "mean-ablation")
RUN_FILES = ("run.json", "history.json", "validation.json", "best.pt")
COMPARISONS = (
    ("product_vs_concat", "product", "concat"),
    ("product_vs_mean", "product", "mean-ablation"),
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _input_identity(manifest, caches):
    rows = load_manifest(manifest)
    return rows, {
        "manifest_sha256": manifest_digest(rows),
        "caches": {name: load_cache(caches[name], rows, name)[1] for name in ("image", "title")},
    }


def train_suite(manifest, caches, config, seeds, output, device):
    rows, inputs = _input_identity(manifest, caches)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "manifest.csv", tuple(rows[0]), rows)
    runs = []
    for seed in seeds:
        for variant in VARIANTS:
            settings = {**config, "model": variant, "seed": seed}
            run = train(manifest, caches, settings, output, f"{variant}-{seed}", device)
            identity = _read_json(run / "run.json")
            if any(identity[key] != value for key, value in inputs.items()):
                raise ValueError("Inputs changed during suite training")
            validation = _read_json(run / "validation.json")
            history = _read_json(run / "history.json")
            runs.append({
                "seed": seed, "model": variant, "directory": run.name,
                "validation": {key: validation[key] for key in ("epoch", "macro_f1", "accuracy")},
                "head_parameters": identity["trainable_head_parameters"],
                "fit_loop_seconds": sum(epoch["epoch_seconds"] for epoch in history),
                "files": {name: _digest(run / name) for name in RUN_FILES},
            })
    for seed in seeds:
        parameters = {run["model"]: run["head_parameters"] for run in runs if run["seed"] == seed}
        if parameters["product"] != parameters["mean-ablation"]:
            raise ValueError("Product and mean-ablation heads must have the same parameter count")
    plan = {
        "schema": 1, "config": config, "seeds": seeds, "training_device": str(device),
        **inputs, "runs": runs,
    }
    path = output / "suite.json"
    _write_json(path, plan)
    return path


def _verified_test(test_dir, expected, best_epoch, checkpoint_sha256):
    metrics = _read_json(test_dir / "metrics.json")
    predictions = read_csv(test_dir / "predictions.csv")
    by_id = {row["sample_id"]: row["label"] for row in predictions}
    if by_id != expected or len(predictions) != len(expected):
        raise ValueError(f"Test prediction membership or labels differ: {test_dir}")
    if any(row["prediction"] not in LABEL_IDS for row in predictions):
        raise ValueError(f"Unknown predicted class: {test_dir}")
    target = torch.tensor([LABEL_IDS[row["label"]] for row in predictions])
    pred = torch.tensor([LABEL_IDS[row["prediction"]] for row in predictions])
    recomputed = classification_metrics(target, pred)
    if (metrics["best_epoch"] != best_epoch or metrics["split"] != "test"
            or metrics.get("checkpoint_sha256") != checkpoint_sha256
            or any(metrics[key] != value for key, value in recomputed.items())):
        raise ValueError(f"Test metrics differ from predictions or checkpoint: {test_dir}")
    return metrics


def _summary(records):
    by_variant = {
        variant: [record["test_macro_f1"] for record in records if record["model"] == variant]
        for variant in VARIANTS
    }
    paired = {}
    seeds = [record["seed"] for record in records if record["model"] == VARIANTS[0]]
    for name, left, right in COMPARISONS:
        deltas = [a - b for a, b in zip(by_variant[left], by_variant[right], strict=True)]
        paired[name] = {
            "by_seed": [{"seed": seed, "macro_f1_delta": delta}
                        for seed, delta in zip(seeds, deltas, strict=True)],
            "mean_delta": mean(deltas),
            "sample_sd": stdev(deltas) if len(deltas) > 1 else None,
        }
    return {
        "runs": records,
        "macro_f1": {
            variant: {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}
            for variant, values in by_variant.items()
        },
        "paired": paired,
    }


def test_suite(manifest, caches, output, device):
    output = Path(output)
    plan = _read_json(output / "suite.json")
    rows, inputs = _input_identity(manifest, caches)
    if plan.get("schema") != 1 or any(plan[key] != value for key, value in inputs.items()):
        raise ValueError("Suite inputs differ from the completed training runs")
    expected_runs = [(seed, variant) for seed in plan["seeds"] for variant in VARIANTS]
    if not expected_runs or [(item["seed"], item["model"]) for item in plan["runs"]] != expected_runs:
        raise ValueError("Suite membership is incomplete or out of order")
    expected = {row["sample_id"]: row["label"] for row in rows if row["split"] == "test"}
    for item in plan["runs"]:
        name = f"{item['model']}-{item['seed']}"
        run = output / name
        if name != item["directory"] or item["files"] != {key: _digest(run / key) for key in RUN_FILES}:
            raise ValueError(f"Training artifacts changed: {name}")
    records = []
    for item in plan["runs"]:
        run = output / item["directory"]
        test_dir = run / "test"
        if not test_dir.exists():
            evaluate_test(manifest, caches, run, device)
        metrics = _verified_test(
            test_dir, expected, item["validation"]["epoch"], item["files"]["best.pt"],
        )
        if metrics["device"] != str(device):
            raise ValueError(f"Test device differs from the suite test request: {run.name}")
        records.append({
            "model": item["model"], "seed": item["seed"], "directory": item["directory"],
            "best_epoch": item["validation"]["epoch"],
            "validation_macro_f1": item["validation"]["macro_f1"],
            "test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"],
            "test_samples": len(expected), "head_parameters": item["head_parameters"],
            "fit_loop_seconds": item["fit_loop_seconds"],
            "test_metrics_sha256": _digest(test_dir / "metrics.json"),
            "test_predictions_sha256": _digest(test_dir / "predictions.csv"),
        })
    path = output / "comparison.json"
    _write_json(path, {"schema": 1, "suite_sha256": _digest(output / "suite.json"),
                       "test_device": str(device), **_summary(records)})
    return path
