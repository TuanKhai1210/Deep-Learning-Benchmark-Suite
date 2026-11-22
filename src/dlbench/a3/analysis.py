"""Recompute ABO results from locked checkpoints and saved predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory

from .data import CLASSES, load_manifest, manifest_digest, read_csv, write_csv
from .fusion_suite import _digest, _read_json, _summary, verified_record, verify_plan


def _averages(values):
    return {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}


def _load_results(suite):
    plan = _read_json(suite / "suite.json")
    variants = verify_plan(suite, plan, team=True)
    result = _read_json(suite / "comparison.json")
    rows = load_manifest(suite / "manifest.csv")
    if (result.get("schema") != 1 or result["suite_sha256"] != _digest(suite / "suite.json")
            or manifest_digest(rows) != plan["manifest_sha256"]
            or result["unimodal_reference"] != plan["unimodal_reference"]):
        raise ValueError("Comparison, validation plan and manifest do not match")
    expected = {row["sample_id"]: row["label"] for row in rows if row["split"] == "test"}
    records, details, predictions = [], {}, {}
    for item in plan["runs"]:
        key = item["seed"], item["model"]
        run = suite / item["directory"]
        record, scores = verified_record(run, item, expected, result["test_device"])
        records.append(record)
        pred_path = run / "test" / "predictions.csv"
        details[key] = scores
        predictions[key] = {row["sample_id"]: row["prediction"] for row in read_csv(pred_path)}
    for key, value in _summary(records, variants, plan["unimodal_reference"]).items():
        if result[key] != value:
            raise ValueError(f"Comparison {key} disagrees with saved test artifacts")
    return plan, result, variants, rows, details, predictions


def _plots(output, suite, result, variants, seeds, details):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    values = [result["macro_f1"][model] for model in variants]
    axis.bar(variants, [100 * value["mean"] for value in values],
             yerr=[100 * (value["sample_sd"] or 0) for value in values], capsize=5, color="#28798d")
    axis.set(ylim=(0, 105), ylabel="Test macro-F1 (%)", title="Mean across seeds ± sample SD")
    fig.savefig(output / "test_macro_f1.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(variants), figsize=(17, 3.6), layout="constrained", sharey=True)
    for axis, model in zip(axes, variants):
        for seed in seeds:
            history = _read_json(suite / f"{model}-{seed}" / "history.json")
            axis.plot([row["epoch"] for row in history], [100 * row["validation_macro_f1"]
                      for row in history], marker=".", label=str(seed))
        axis.set(title=model, xlabel="Epoch", ylim=(0, 100))
    axes[0].set_ylabel("Validation macro-F1 (%)")
    axes[-1].legend(title="Seed")
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained", sharex=True, sharey=True)
    rendered = []
    for axis, model in zip(axes.flat, variants):
        counts = np.asarray([details[seed, model]["confusion_matrix"] for seed in seeds])
        fractions = (counts / counts.sum(axis=2, keepdims=True)).mean(axis=0)
        rendered.append(axis)
        heatmap = axis.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        axis.set(title=model, xlabel="Predicted class", ylabel="True class")
        axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=55, ha="right", fontsize=7)
        axis.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
        axis.tick_params(labelbottom=True, labelleft=True)
    for axis in list(axes.flat)[len(variants):]:
        axis.set_axis_off()
    fig.colorbar(heatmap, ax=rendered, shrink=0.7, label="Fraction of true class")
    fig.savefig(output / "test_confusions.png", dpi=160)
    plt.close(fig)


def analyze(suite, output):
    suite, output = Path(suite), Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan, result, variants, rows, details, predictions = _load_results(suite)
    seeds = plan["seeds"]
    by_model = {}
    per_class = []
    for model in variants:
        runs = [run for run in result["runs"] if run["model"] == model]
        by_model[model] = {
            key: _averages([run[key] for run in runs])
            for key in ("validation_macro_f1", "test_macro_f1", "test_accuracy",
                        "fit_loop_seconds", "cached_test_seconds", "cached_test_samples_per_second")
            if all(run[key] is not None for run in runs)
            and (key != "fit_loop_seconds" or not plan.get("distributed"))
        }
        for index, label in enumerate(CLASSES):
            scores = [details[seed, model]["per_class"][index] for seed in seeds]
            per_class.append({"model": model, "label": label, "test_support": scores[0]["support"],
                              "mean_f1": mean(score["f1"] for score in scores),
                              "sample_sd_f1": stdev(score["f1"] for score in scores)
                              if len(scores) > 1 else None})
    errors = []
    agreement = defaultdict(lambda: {"samples": 0, "concat_correct": 0, "product_correct": 0})
    for seed in seeds:
        for row in sorted(rows, key=lambda row: row["sample_id"]):
            if row["split"] != "test":
                continue
            sample, label = row["sample_id"], row["label"]
            guesses = {model: predictions[seed, model][sample] for model in variants}
            image_correct = guesses["image-only"] == label
            title_correct = guesses["title-only"] == label
            category = ("both_correct" if image_correct and title_correct else
                        "image_only_correct" if image_correct else
                        "title_only_correct" if title_correct else
                        "agree_on_error" if guesses["image-only"] == guesses["title-only"] else
                        "different_errors")
            bucket = agreement[seed, category]
            bucket["samples"] += 1
            for model in ("concat", "product"):
                bucket[f"{model}_correct"] += guesses[model] == label
            for name, pair in result["paired"].items():
                left, right = pair["left"], pair["right"]
                if (guesses[left] == label) != (guesses[right] == label):
                    errors.append({
                        "comparison": name, "seed": seed, "sample_id": sample, "label": label,
                        "change": "corrected" if guesses[left] == label else "introduced_error",
                        **guesses, "title": row["title"], "image_path": row["image_path"],
                    })

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        folder = Path(staging) / "analysis"
        folder.mkdir()
        score_rows = []
        for run in result["runs"]:
            row = {key: value for key, value in run.items() if key != "training_profile"}
            if "training_profile" in run:
                row.update({f"training_{key}": value for key, value in run["training_profile"].items()})
            score_rows.append(row)
        write_csv(folder / "scores_by_seed.csv", tuple(score_rows[0]), score_rows)
        write_csv(folder / "per_class.csv", tuple(per_class[0]), per_class)
        write_csv(folder / "error_cases.csv",
                  ("comparison", "seed", "sample_id", "label", "change", *variants, "title", "image_path"),
                  errors)
        write_csv(folder / "modality_agreement.csv",
                  ("seed", "category", "samples", "concat_correct", "product_correct"), (
                      {"seed": seed, "category": category, **counts}
                      for (seed, category), counts in sorted(agreement.items())
                  ))
        write_csv(folder / "paired_deltas.csv", ("comparison", "left", "right", "seed", "macro_f1_delta"), (
            {"comparison": name, "left": pair["left"], "right": pair["right"], **record}
            for name, pair in result["paired"].items() for record in pair["by_seed"]
        ))
        (folder / "summary.json").write_text(json.dumps({
            "schema": 1, "manifest_sha256": plan["manifest_sha256"],
            "comparison_sha256": _digest(suite / "comparison.json"),
            "seeds": seeds, "training_device": plan["training_device"], "test_device": result["test_device"],
            "distributed": plan.get("distributed", False),
            "unimodal_reference": plan["unimodal_reference"], "by_model": by_model,
            "paired": result["paired"], "encoders": {key: value["encoder"] for key, value in plan["caches"].items()},
            "cached_test_timing_scope": "Held-out scoring loop with cached vectors, transfers, loss and metrics",
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _plots(folder, suite, result, variants, seeds, details)
        folder.rename(output)
    return output
