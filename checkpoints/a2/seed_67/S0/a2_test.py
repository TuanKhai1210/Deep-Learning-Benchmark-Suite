"""Evaluate fixed test data once after main completion; never retrain/select on test."""
from pathlib import Path
import csv
import hashlib
import json


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def selected_run(root):
    root = Path(root).resolve()
    pointer = root / "resume_pointer.json"
    if pointer.is_file():
        value = json.loads(pointer.read_text(encoding="utf-8"))
        target = (root / value["target_relative"]).resolve()
        if not target.is_relative_to(root / "continuations"):
            raise ValueError("Resume pointer escapes the experiment")
        metadata = json.loads((target / "run/metadata.json").read_text(encoding="utf-8"))
        if metadata["identity_sha256"] != value["identity_sha256"]:
            raise ValueError("Resume pointer identity mismatch")
        return target
    return root


def assert_complete(state, config, metadata):
    from dlbench.a2.data import content_hash
    from dlbench.a2.config import validate_config
    validate_config(config, main=True)
    if metadata["kind"] != "main" or metadata["identity"]["development"]:
        raise ValueError("Test requires a main checkpoint")
    if metadata["identity"]["config"] != config:
        raise ValueError("Saved effective config differs from run identity")
    if content_hash(metadata["identity"]) != metadata["identity_sha256"]:
        raise ValueError("Metadata identity hash mismatch")
    if state["identity_sha256"] != metadata["identity_sha256"]:
        raise ValueError("Last checkpoint identity mismatch")
    epoch = state["epoch"]
    if type(epoch) is not int or not 1 <= epoch <= config["budget"]["epochs"]:
        raise ValueError("Invalid completed epoch")
    if len(state["history"]) != epoch or state["best"] is None:
        raise ValueError("Incomplete checkpoint history/selection")
    completed = epoch == config["budget"]["epochs"]
    early_stopped = state["bad_epochs"] >= config["budget"]["patience"]
    if not (completed or early_stopped):
        raise ValueError(f"Main run is only at epoch {epoch}; resume training before test evaluation")
    return "budget_completed" if completed else "early_stopping"


def run_test(root, data_root, manifest, *, device="cuda"):
    import torch
    from dlbench.a2.data import TrashCanDataset, content_hash
    from dlbench.a2.trainer import evaluate_checkpoint, save_json, source_identity
    root = selected_run(root)
    run = root / "run"
    config = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    stop_reason = assert_complete(state, config, metadata)
    selected_epoch = state["best"]["epoch"]
    best_state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if (best_state["epoch"] != selected_epoch
            or best_state["identity_sha256"] != metadata["identity_sha256"]):
        raise ValueError("best.pt differs from validation selection; repair checkpoint before test")
    del best_state
    if source_identity()["source_digest"] != metadata["identity"]["source_digest"]:
        raise ValueError("Current model/evaluator source differs from the run snapshot")
    del state
    dataset = TrashCanDataset(data_root, manifest, "test", config["preprocessing"],
                              seed=config["run"]["seed"], color_jitter=False)
    if len(dataset) != 1078:
        raise ValueError("Expected all 1078 fixed test images")
    checkpoint_sha = file_hash(run / "best.pt")
    membership = content_hash(dataset.rows)
    identity = {"checkpoint_sha256": checkpoint_sha,
                "run_identity_sha256": metadata["identity_sha256"], "membership_sha256": membership}
    output = root / "test_best.json"
    marker = root / "test_complete.json"
    if output.is_file() and marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["identity"] == identity and record["output_sha256"] == file_hash(output):
            result = json.loads(output.read_text(encoding="utf-8"))
            print("Reusing matching verified test result; no inference or training repeated.")
        else:
            raise ValueError("Existing test result/cache identity changed; preserve it for review")
    else:
        # Core evaluator loads selected best.pt; no test score affects selection.
        result = evaluate_checkpoint(run, dataset, device=device, workers=0)
        result["training_stop_reason"] = stop_reason
        result["parameter_counts"] = metadata["parameter_counts"]
        from dlbench.common.reproducibility import collect_environment
        result["evaluation_environment"] = collect_environment()
        result["evaluation_device"] = str(device)
        if any(result[key] != value for key, value in identity.items()):
            raise ValueError("Test evaluator provenance differs from selected checkpoint")
        save_json(output, result)
        save_json(marker, {"identity": identity, "output_sha256": file_hash(output)})
    if result["split"] != "test":
        raise ValueError("Cached output is not the test partition")
    if result["selected_epoch"] != selected_epoch:
        raise ValueError("Test output selected epoch differs from validation selection")
    metrics = result["metrics"]
    with (root / "test_per_class.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("id", "name", "iou", "dice", "support", "predicted")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    with (root / "test_confusion.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(metrics["confusion_matrix"])
    print("TEST foreground mIoU:", metrics["foreground_miou"])
    print("TEST foreground Dice:", metrics["foreground_dice"])
    print("TEST trash-only mIoU:", metrics["trash_miou"])
    print("Saved:", output)
    return root, result


def launch(namespace):
    required = ("RUN_ROOT", "DATA_ROOT", "MANIFEST")
    if any(key not in namespace for key in required):
        raise ValueError("Run notebook setup/data cells first")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root, result = run_test(namespace["RUN_ROOT"], namespace["DATA_ROOT"], namespace["MANIFEST"], device=device)
    namespace["RUN_ROOT"], namespace["best_test"] = root, result
