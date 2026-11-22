"""Run matched ABO experiments on one shared cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from .data import load_manifest, manifest_digest, read_csv, write_csv
from .engine import LABEL_IDS, _write_json, classification_metrics, evaluate_test, train
from .features import load_cache
from .model import MODALITIES

VARIANTS = ("concat", "product", "mean-ablation")
TEAM_VARIANTS = tuple(MODALITIES)
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


def _unimodal_reference(runs):
    # Exact ties use declaration order, so image-only wins a tie.
    scores = {model: mean(run["validation"]["macro_f1"] for run in runs if run["model"] == model)
              for model in ("image-only", "title-only")}
    return {"model": max(scores, key=scores.__getitem__), "mean_validation_macro_f1": scores}


def training_record(run, inputs, config, model, seed):
    identity = _read_json(run / "run.json")
    expected_caches = {name: inputs["caches"][name] for name in MODALITIES[model]}
    if (identity["manifest_sha256"] != inputs["manifest_sha256"]
            or identity["caches"] != expected_caches
            or identity["config"] != {**config, "model": model, "seed": seed}):
        raise ValueError(f"Run inputs or settings differ: {run.name}")
    validation = _read_json(run / "validation.json")
    history = _read_json(run / "history.json")
    if [epoch["epoch"] for epoch in history] != list(range(1, config["epochs"] + 1)):
        raise ValueError(f"Run did not complete the full training budget: {run.name}")
    best = max(history, key=lambda epoch: epoch["validation_macro_f1"])
    if any(validation[key] != best[history_key] for key, history_key in (
            ("epoch", "epoch"), ("macro_f1", "validation_macro_f1"),
            ("accuracy", "validation_accuracy"))):
        raise ValueError(f"Run validation does not match the selected epoch: {run.name}")
    return {
        "seed": seed, "model": model, "directory": run.name,
        "validation": {key: validation[key] for key in ("epoch", "macro_f1", "accuracy")},
        "head_parameters": identity["trainable_head_parameters"],
        "fit_loop_seconds": sum(epoch["epoch_seconds"] for epoch in history),
        "files": {name: _digest(run / name) for name in RUN_FILES},
    }


def verify_matched_heads(runs, seeds):
    for seed in seeds:
        parameters = {run["model"]: run["head_parameters"] for run in runs if run["seed"] == seed}
        if parameters["product"] != parameters["mean-ablation"]:
            raise ValueError("Product and mean-ablation heads must have the same parameter count")


def train_suite(manifest, caches, config, seeds, output, device, *, team=False):
    variants = TEAM_VARIANTS if team else VARIANTS
    rows, inputs = _input_identity(manifest, caches)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "manifest.csv", tuple(rows[0]), rows)
    runs = []
    for seed in seeds:
        for variant in variants:
            settings = {**config, "model": variant, "seed": seed}
            run = train(manifest, caches, settings, output, f"{variant}-{seed}", device)
            runs.append(training_record(run, inputs, config, variant, seed))
    verify_matched_heads(runs, seeds)
    plan = {
        "schema": 1, "config": config, "seeds": seeds, "training_device": str(device),
        **inputs, "runs": runs,
    }
    if team:
        plan.update({
            "profile": "team",
            "unimodal_reference": _unimodal_reference(runs),
            "source_sha256": {path.name: _digest(path)
                              for path in sorted(Path(__file__).parent.glob("*.py"))},
        })
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


def _summary(records, variants=VARIANTS, reference=None):
    comparisons = list(COMPARISONS)
    if reference:
        comparisons.insert(0, ("concat_vs_best_unimodal", "concat", reference["model"]))
    by_variant = {
        variant: [record["test_macro_f1"] for record in records if record["model"] == variant]
        for variant in variants
    }
    paired = {}
    seeds = [record["seed"] for record in records if record["model"] == VARIANTS[0]]
    for name, left, right in comparisons:
        deltas = [a - b for a, b in zip(by_variant[left], by_variant[right], strict=True)]
        paired[name] = {
            "by_seed": [{"seed": seed, "macro_f1_delta": delta}
                        for seed, delta in zip(seeds, deltas, strict=True)],
            "left": left, "right": right,
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


def verify_plan(output, plan, *, team):
    variants = TEAM_VARIANTS if team else VARIANTS
    profile = "team" if team else "fusion"
    if plan.get("schema") != 1 or plan.get("profile", "fusion") != profile:
        raise ValueError(f"Expected a completed {profile} suite")
    seeds = plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Suite must contain distinct run seeds")
    expected_runs = [(seed, variant) for seed in plan["seeds"] for variant in variants]
    if not expected_runs or [(item["seed"], item["model"]) for item in plan["runs"]] != expected_runs:
        raise ValueError("Suite membership is incomplete or out of order")
    for item in plan["runs"]:
        name = f"{item['model']}-{item['seed']}"
        run = output / name
        if name != item["directory"] or item["files"] != {key: _digest(run / key) for key in RUN_FILES}:
            raise ValueError(f"Training artifacts changed: {name}")
        validation = _read_json(run / "validation.json")
        if item["validation"] != {key: validation[key] for key in ("epoch", "macro_f1", "accuracy")}:
            raise ValueError(f"Suite validation scores disagree with training artifacts: {name}")
    if team and plan.get("unimodal_reference") != _unimodal_reference(plan["runs"]):
        raise ValueError("Unimodal reference disagrees with validation scores")
    return variants


def verified_record(run, item, expected, device):
    test_dir = run / "test"
    metrics = _verified_test(test_dir, expected, item["validation"]["epoch"], item["files"]["best.pt"])
    if metrics["device"] != device:
        raise ValueError(f"Test device differs from the suite test request: {run.name}")
    return {
        **{key: item[key] for key in ("batch_id", "training_profile") if key in item},
        "model": item["model"], "seed": item["seed"], "directory": item["directory"],
        "best_epoch": item["validation"]["epoch"],
        "validation_macro_f1": item["validation"]["macro_f1"],
        "test_macro_f1": metrics["macro_f1"], "test_accuracy": metrics["accuracy"],
        "test_samples": len(expected), "head_parameters": item["head_parameters"],
        "fit_loop_seconds": item["fit_loop_seconds"],
        "cached_test_seconds": metrics.get("cached_test_seconds"),
        "cached_test_samples_per_second": metrics.get("cached_test_samples_per_second"),
        "test_metrics_sha256": _digest(test_dir / "metrics.json"),
        "test_predictions_sha256": _digest(test_dir / "predictions.csv"),
    }, metrics


def test_suite(manifest, caches, output, device, *, team=False):
    output = Path(output)
    plan = _read_json(output / "suite.json")
    variants = verify_plan(output, plan, team=team)
    rows, inputs = _input_identity(manifest, caches)
    if any(plan[key] != value for key, value in inputs.items()):
        raise ValueError("Suite inputs differ from the completed training runs")
    expected = {row["sample_id"]: row["label"] for row in rows if row["split"] == "test"}
    existing = {}
    for item in plan["runs"]:
        run = output / item["directory"]
        if (run / "test").exists():
            existing[run.name], _ = verified_record(run, item, expected, str(device))
    records = []
    for item in plan["runs"]:
        run = output / item["directory"]
        if run.name not in existing:
            evaluate_test(manifest, caches, run, device)
            existing[run.name], _ = verified_record(run, item, expected, str(device))
        records.append(existing[run.name])
    path = output / "comparison.json"
    reference = plan.get("unimodal_reference")
    summary = _summary(records, variants, reference)
    _write_json(path, {"schema": 1, "suite_sha256": _digest(output / "suite.json"),
                       "test_device": str(device), "unimodal_reference": reference, **summary})
    return path
