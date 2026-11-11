"""Prospective augmentation rule, using validation only and a matched pair."""
from copy import deepcopy
from .data import content_hash


def color_decision(off_metadata, on_metadata, off_validation, on_validation):
    configs = [deepcopy(m["identity"]["config"]) for m in (off_metadata, on_metadata)]
    if configs[0]["run"]["color_jitter"] or not configs[1]["run"]["color_jitter"]:
        raise ValueError("Expected S0 off and S1 on")
    for config in configs:
        if config["model"]["name"] != "segformer":
            raise ValueError("Color comparison is within SegFormer")
        config["run"].pop("name")
        config["run"].pop("color_jitter")
    if configs[0] != configs[1]:
        raise ValueError("Color pair has different settings")
    for field in ("initial_state_sha256",):
        if off_metadata[field] != on_metadata[field]:
            raise ValueError(f"Pair mismatch: {field}")
    for field in ("source_digest", "device", "environment", "dependencies", "cpu_threads",
                  "train_membership", "validation_membership", "development"):
        if off_metadata["identity"][field] != on_metadata["identity"][field]:
            raise ValueError(f"Pair mismatch: {field}")
    for metadata, record in zip((off_metadata, on_metadata), (off_validation, on_validation)):
        if metadata["identity"]["development"] or record["kind"] != "main":
            raise ValueError("Final color decisions require main-run validation evidence")
        if content_hash(metadata["identity"]) != metadata["identity_sha256"]:
            raise ValueError("Run identity hash mismatch")
        if record["run_identity_sha256"] != metadata["identity_sha256"]:
            raise ValueError("Evaluation belongs to a different run")
        if record["membership_sha256"] != metadata["identity"]["validation_membership"]:
            raise ValueError("Evaluation membership differs from the paired validation split")
        if record["split"] != "validation":
            raise ValueError("Augmentation decisions require validation, never test")
    deltas = {}
    for metric in ("trash_miou", "foreground_miou", "foreground_dice"):
        off, on = off_validation["metrics"][metric], on_validation["metrics"][metric]
        if off is None or on is None:
            raise ValueError("Undefined metric in color comparison")
        deltas[metric] = on - off
    return {"validation_deltas": deltas,
            "accept_color": deltas["trash_miou"] >= 0.01 - 1e-12
            and deltas["foreground_miou"] >= -0.01 - 1e-12
            and deltas["foreground_dice"] >= -0.01 - 1e-12}
