"""Owner A (Thiên) for error plots; B for comparisons; C for report integration."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
import csv
import hashlib
import io
import json
import math
from dlbench.a1.contracts import Predictions
from dlbench.common.artifacts import HISTORY_COLUMNS, _validate_history_row


def save_predictions(predictions: Predictions, path: Path) -> None:
    """CSV: sample_id, target, predicted_label, then probability_0..probability_9."""
    count = len(predictions.sample_ids)
    if not count or any(len(values) != count for values in (
        predictions.targets, predictions.predicted_labels, predictions.probabilities
    )):
        raise ValueError("Prediction columns must have equal nonzero lengths")
    if any(not isinstance(s, str) or not s for s in predictions.sample_ids):
        raise ValueError("Sample IDs must be nonempty strings")
    if len(set(predictions.sample_ids)) != count:
        raise ValueError("Sample IDs must be unique")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["sample_id", "target", "predicted_label"] + [f"probability_{i}" for i in range(10)])
    for sample_id, target, label, probabilities in zip(
        predictions.sample_ids, predictions.targets, predictions.predicted_labels, predictions.probabilities
    ):
        if any(type(v) is not int or not 0 <= v < 10 for v in (target, label)):
            raise ValueError("Labels must be integers in 0..9")
        if len(probabilities) != 10 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError("Expected ten finite probabilities in [0, 1]")
        if not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        if probabilities[label] != max(probabilities):
            raise ValueError("Predicted label must maximize probability")
        writer.writerow([sample_id, target, label, *probabilities])
    with Path(path).open("x", encoding="utf-8", newline="") as file:
        file.write(output.getvalue())


def _read_run(run_dir: Path):
    config, metadata, metrics = [
        json.loads((run_dir / name).read_text(encoding="utf-8"))
        for name in ("config.json", "metadata.json", "metrics.json")
    ]
    if metrics.get("eval_split") != "validation":
        raise ValueError("Learning curves require validation metrics")
    text = (run_dir / "history.csv").read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise ValueError("History has an incomplete final line")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != list(HISTORY_COLUMNS):
        raise ValueError("Unexpected history columns")
    rows = []
    for saved in reader:
        if set(saved) != set(HISTORY_COLUMNS) or any(v is None for v in saved.values()):
            raise ValueError("Malformed history row")
        row = {k: int(v) if k == "epoch" else float(v) for k, v in saved.items()}
        _validate_history_row(row)
        if row["epoch"] != len(rows):
            raise ValueError("History epochs must be contiguous starting at zero")
        rows.append(row)
    if not rows:
        raise ValueError("History is empty")
    best = min(rows, key=lambda r: (-r["val_macro_f1"], r["val_loss"], r["epoch"]))
    if best["epoch"] != metrics["epoch"] or any(
        not math.isclose(best[key], metrics[key], rel_tol=1e-7, abs_tol=1e-9)
        for key in ("val_loss", "val_accuracy", "val_macro_f1")
    ):
        raise ValueError("Saved best metrics disagree with history")
    if metadata["model_name"] != config["model"]["name"] or metadata["run_seed"] != config["run"]["seed"]:
        raise ValueError("Config and metadata identity disagree")
    for record in metadata["sources"].values():
        snapshot = (run_dir / record["snapshot"]).resolve()
        if not snapshot.is_relative_to(run_dir.resolve()):
            raise ValueError("Source snapshot escapes run directory")
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Source snapshot hash mismatch")
    return config, metadata, metrics, rows


def analyze_run(run_dir: Path, *, output_dir: Path | None = None) -> None:
    """Use saved history/predictions; do not retrain or pick a new checkpoint.

    TODO A (Thiên): fixed-order confusion matrix, learning curves, correct/error samples.
    TODO C (Khải): preserve run provenance and explicit train/val/test labels on plots.
    Only figures selected after review should be copied to docs/assets/a1.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    run_dir = Path(run_dir)
    config, metadata, metrics, rows = _read_run(run_dir)
    destination = Path(output_dir) if output_dir is not None else run_dir / "analysis"
    summary = {
        "run_id": run_dir.name, "source_run": str(run_dir.resolve()),
        "model": metadata["model_name"], "seed": metadata["run_seed"],
        "run_mode": metadata["run_mode"], "eval_split": "validation",
        "git_revision": metadata["git_revision"], "git_dirty": metadata["git_dirty"],
        "split_hash": metadata["split"]["sha256"],
        "statistics_hash": metadata["normalization"]["sha256"],
        "completed_epochs": len(rows), "best_epoch_zero_based": metrics["epoch"],
        "metrics": metrics, "training_config": config.get("training"),
        "epoch_seconds_sum": sum(r["epoch_seconds"] for r in rows),
        "timing_note": "Sum of recorded epoch times; not inference timing or complete fit wall time.",
        "artifact_sha256": {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
                            for name in ("config.json", "metadata.json", "metrics.json", "history.csv")},
        "limitations": "Single-run validation results. No test claims or multi-seed uncertainty. Confusion/error plots require predictions.",
    }
    figure = Figure(figsize=(11, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axis, metric, title in zip(figure.subplots(1, 2), ("loss", "macro_f1"), ("Loss", "Macro F1")):
        for split, label in (("train", "Train"), ("val", "Validation")):
            axis.plot([r["epoch"] + 1 for r in rows], [r[f"{split}_{metric}"] for r in rows], label=label)
        axis.axvline(metrics["epoch"] + 1, color="gray", linestyle="--", label="Saved best epoch")
        axis.set(xlabel="Epoch (1-based)", ylabel=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"{metadata['model_name']} | seed {metadata['run_seed']} | {metadata['run_mode']} | validation")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    figure.savefig(destination / "learning_curves.png", dpi=160)


def compare_runs(run_dirs: Sequence[Path], output_path: Path) -> None:
    """Export compatible per-run validation rows, without pooling unlike configs.

    This table deliberately makes no multi-seed mean/std or hardware speed claim.
    """
    if not run_dirs or len({Path(p).resolve() for p in run_dirs}) != len(run_dirs):
        raise ValueError("Supply distinct nonempty run directories")
    loaded = [(Path(p), _read_run(Path(p))) for p in run_dirs]
    signatures = {
        json.dumps([m["protocol_id"], m["split"]["sha256"], m["normalization"]["sha256"],
                    m["run_mode"], c["preprocessing"], c["evaluation"], c["timing"]], sort_keys=True)
        for _, (c, m, _, _) in loaded
    }
    if len(signatures) != 1:
        raise ValueError("Runs have incompatible protocol, split, preprocessing, evaluation or timing settings")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["run_id", "model", "seed", "eval_split", "git_dirty", "epochs", "best_epoch_zero_based",
                     "val_loss", "val_accuracy", "val_macro_f1", "training_config"])
    for path, (config, metadata, metrics, rows) in loaded:
        writer.writerow([path.name, metadata["model_name"], metadata["run_seed"], "validation", metadata["git_dirty"],
                         len(rows), metrics["epoch"], metrics["val_loss"], metrics["val_accuracy"], metrics["val_macro_f1"],
                         json.dumps(config.get("training"), sort_keys=True)])
    with Path(output_path).open("x", encoding="utf-8", newline="") as file:
        file.write(output.getvalue())


def analyze_predictions(predictions: Predictions, output_dir: Path, *, split: str) -> None:
    """Fixed ten-class confusion counts plus stable IDs of misclassified samples."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if split not in ("validation", "test"):
        raise ValueError("Specify validation or test explicitly")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_predictions(predictions, output_dir / f"predictions_{split}.csv")
    matrix = np.zeros((10, 10), dtype=int)
    errors = []
    for sample_id, target, predicted, probabilities in zip(
        predictions.sample_ids, predictions.targets, predictions.predicted_labels, predictions.probabilities
    ):
        matrix[target, predicted] += 1
        if target != predicted:
            errors.append({"sample_id": sample_id, "target": target, "predicted_label": predicted,
                           "confidence": probabilities[predicted]})
    figure = Figure(figsize=(8, 7), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    plotted = axis.imshow(matrix, cmap="Blues")
    names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    axis.set(xticks=range(10), yticks=range(10), xticklabels=names, yticklabels=names,
             xlabel="Predicted class", ylabel="True class", title=f"{split.capitalize()} confusion matrix (counts)")
    axis.tick_params(axis="x", labelrotation=60)
    for i in range(10):
        for j in range(10):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=8)
    figure.colorbar(plotted, ax=axis)
    figure.savefig(output_dir / f"confusion_{split}.png", dpi=160)
    payload = {"eval_split": split, "labels": list(range(10)), "class_names": names,
               "num_samples": len(predictions.targets), "confusion_counts": matrix.tolist(),
               "errors": sorted(errors, key=lambda e: (-e["confidence"], e["sample_id"]))}
    (output_dir / "errors.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
