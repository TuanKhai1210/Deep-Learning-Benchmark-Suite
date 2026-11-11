"""Verify and summarize the bundled A2 test evidence with Python alone."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("U", "S0", "S1")
SEEDS = (69420, 67, 69)
METRICS = ("foreground_miou", "foreground_dice", "trash_miou")
MANIFEST_DIGEST = "87905f1a97dd2c04194e049ca61d1a7f3bbe2942e9fe17dc2d737c371cdce943"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(actual, expected, context):
    check(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0), context)


def content_hash(value):
    contents = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(contents).hexdigest()


def verify_manifest():
    manifest = read(ROOT / "evidence/manifest.json")
    check(manifest.pop("content_sha256") == content_hash(manifest) == MANIFEST_DIGEST,
          "Canonical split manifest digest")
    check(manifest["split_seed"] == 36, "Split seed")
    rows = manifest["images"]
    check(Counter(row["split"] for row in rows) ==
          {"train": 5057, "validation": 1077, "test": 1078}, "Split sizes")
    videos = {}
    for row in rows:
        check(videos.setdefault(row["video_id"], row["split"]) == row["split"], "Video leakage")
    return content_hash([row for row in rows if row["split"] == "test"])


def verify_run(row):
    run = ROOT / row["artifact_key"]
    contents = (run / "test_best.json").read_bytes()
    identity = {key: row[key] for key in (
        "checkpoint_sha256", "run_identity_sha256", "membership_sha256")}
    check(hashlib.sha256(contents).hexdigest() == row["test_output_sha256"], "Test SHA-256")
    test = json.loads(contents)
    marker = read(run / "test_complete.json")
    check(test["split"] == "test" and test["selected_epoch"] == row["best_epoch"], "Test split/epoch")
    check(all(test[k] == v for k, v in identity.items()), "Test identity")
    check(marker["identity"] == identity and marker["output_sha256"] == row["test_output_sha256"],
          "Test completion marker")
    history = read(run / "history.json")
    config = read(run / "effective_config.json")
    check((config["run"]["name"], config["run"]["seed"]) == (row["model"], row["seed"]), "Config run")
    check([entry["epoch"] for entry in history] == list(range(1, row["epochs"] + 1)), "Epoch sequence")
    best = min(history, key=lambda epoch: (
        -epoch["validation"]["foreground_miou"], epoch["validation"]["loss"], epoch["epoch"]))
    check(best["epoch"] == test["selected_epoch"], "Validation checkpoint selection")

    metrics = test["metrics"]
    confusion = metrics["confusion_matrix"]
    check(len(confusion) == 17 and all(len(line) == 17 for line in confusion), "Matrix dimensions")
    check(all(type(n) is int and n >= 0 for line in confusion for n in line), "Matrix counts")
    support = [sum(line) for line in confusion]
    predicted = [sum(column) for column in zip(*confusion)]
    check(sum(support) == metrics["valid_pixels"], "Valid pixel count")
    iou, dice = [], []
    check(len(metrics["per_class"]) == 17, "Per-class record count")
    for c, entry in enumerate(metrics["per_class"]):
        check((entry["id"], entry["support"], entry["predicted"]) ==
              (c, support[c], predicted[c]), "Class support")
        true_positive = confusion[c][c]
        total = support[c] + predicted[c]
        iou.append(true_positive / (total - true_positive))
        dice.append(2 * true_positive / total)
        equal_number(iou[c], entry["iou"], "Class IoU")
        equal_number(dice[c], entry["dice"], "Class Dice")
    with (run / "test_confusion.csv").open(encoding="utf-8", newline="") as stream:
        check([[int(n) for n in line] for line in csv.reader(stream)] == confusion, "Confusion CSV")
    with (run / "test_per_class.csv").open(encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream))
    check(len(entries) == 17, "Per-class CSV size")
    for c, entry in enumerate(entries):
        check((int(entry["id"]), entry["name"], int(entry["support"]), int(entry["predicted"])) ==
              (c, metrics["per_class"][c]["name"], support[c], predicted[c]), "Class CSV identity")
        equal_number(iou[c], float(entry["iou"]), "Class CSV IoU")
        equal_number(dice[c], float(entry["dice"]), "Class CSV Dice")
    values = dict(zip(METRICS, (mean(iou[1:]), mean(dice[1:]), mean(iou[9:]))))
    for name, value in values.items():
        equal_number(value, metrics[name], f"Saved test {name}")
        equal_number(value, row["test_metrics"][name], f"Report test {name}")
    return values, best["validation"], support, iou, dice, [line[0] for line in confusion]


def verify_class_summary(results):
    with (ROOT / "evidence/per_class_summary.csv").open(encoding="utf-8", newline="") as stream:
        classes = list(csv.DictReader(stream))
    check(len(classes) == 17, "Aggregate class count")
    for c, entry in enumerate(classes):
        check(int(entry["class_id"]) == c and int(entry["test_support_pixels"]) ==
              results["U", 69][2][c], "Aggregate class support")
        for model in MODELS:
            for metric, index in (("iou", 3), ("dice", 4)):
                values = [100 * results[model, seed][index][c] for seed in SEEDS]
                for stat, actual in (("mean", mean(values)), ("sd", stdev(values))):
                    expected = float(entry[f"{model}_{metric}_{stat}_pct"])
                    check(abs(actual - expected) <= 5.1e-7, "Rounded class aggregate")
        paired = 100 * mean(results["S1", s][3][c] - results["S0", s][3][c] for s in SEEDS)
        check(abs(paired - float(entry["S1_minus_S0_iou_mean_pp"])) <= 5.1e-7, "Paired class IoU")
    with (ROOT / "evidence/background_errors_by_seed.csv").open(encoding="utf-8", newline="") as stream:
        errors = list(csv.DictReader(stream))
    keys = {(row["model"], int(row["seed"]), row["true_group"]) for row in errors}
    check(len(errors) == 18 and keys == {(m, s, g) for m, s in results for g in ("foreground", "trash")},
          "Background error rows")
    for row in errors:
        values = results[row["model"], int(row["seed"])]
        start = 1 if row["true_group"] == "foreground" else 9
        support, missed = sum(values[2][start:]), sum(values[5][start:])
        check((int(row["group_support_pixels"]), int(row["predicted_background_pixels"])) ==
              (support, missed), "Background error counts")
        check(abs(100 * missed / support - float(row["predicted_background_pct"])) <= 5.1e-7,
              "Rounded background error percentage")


def main():
    summary = read(ROOT / "verified_results.json")
    rows = summary["rows"]
    results = {(row["model"], row["seed"]): verify_run(row) for row in rows}
    check(len(rows) == 9 and set(results) == {(model, seed) for model in MODELS for seed in SEEDS},
          "Nine model/seed pairs")
    check({row["membership_sha256"] for row in rows} == {verify_manifest()}, "Common test membership")
    check(all(value[2] == results["U", 69][2] for value in results.values()), "Common class supports")
    verify_class_summary(results)

    print("Model   FG mIoU (%)    FG Dice (%)    Trash mIoU (%)")
    for model in MODELS:
        scores = []
        for metric in METRICS:
            values = [results[model, seed][0][metric] for seed in SEEDS]
            avg, sd = mean(values), stdev(values)
            equal_number(avg, summary["aggregate"][model][metric]["mean"], "Three-seed mean")
            equal_number(sd, summary["aggregate"][model][metric]["sd"], "Three-seed SD")
            scores.append(f"{100*avg:5.2f} +/- {100*sd:.2f}")
        print(f"{model:5}   " + "    ".join(scores))
    for seed in SEEDS:
        for index, name in enumerate(("test_delta", "validation_delta")):
            delta = {metric: results["S1", seed][index][metric] - results["S0", seed][index][metric]
                     for metric in METRICS}
            pair = summary["paired"][str(seed)]
            for metric in METRICS:
                equal_number(delta[metric], pair[name][metric], f"{seed} {name} {metric}")
            if name == "validation_delta":
                met = (delta["trash_miou"] >= .01 and delta["foreground_miou"] >= -.01
                       and delta["foreground_dice"] >= -.01)
                check(met == pair["meets_proposal_threshold"], "Proposal threshold")
    print("Verified 9 test outputs, split membership, validation selections, class CSVs and paired deltas.")


if __name__ == "__main__":
    main()
