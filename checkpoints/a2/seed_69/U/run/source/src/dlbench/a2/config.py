"""Literal Python configs with one shared, explicitly draft A2 protocol."""
from copy import deepcopy
from pathlib import Path
import math
import re

from dlbench.common.config import _read_python_config
from .converter import CONVERTER_VERSION
from .data import MANIFEST_SHA256


def load_config(path):
    path = Path(path).resolve()
    experiment = _read_python_config(path)
    if set(experiment) != {"protocol_file", "model", "training", "run"}:
        raise ValueError("Experiment requires protocol_file/model/training/run only")
    protocol = _read_python_config(path.parent / experiment["protocol_file"])
    if set(protocol) != {"protocol", "data", "preprocessing", "loss", "budget"}:
        raise ValueError("Unexpected shared protocol sections")
    config = deepcopy(protocol)
    config.update({k: deepcopy(experiment[k]) for k in ("model", "training", "run")})
    validate_config(config)
    return config


def validate_config(config, *, main=False):
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    require(set(config) == {"protocol", "data", "preprocessing", "loss", "budget",
                            "model", "training", "run"}, "Unexpected config sections")
    protocol, data = config["protocol"], config["data"]
    require(protocol["version"] == "a2-initial-v1", "Unknown A2 protocol")
    require(data["split_seed"] == 36 and data["manifest_sha256"] == MANIFEST_SHA256,
            "Use the published seed-36 manifest")
    require(data["num_classes"] == 17 and data["ignore_id"] == 255, "Invalid labels")
    require(data["converter"] == CONVERTER_VERSION and data["pillow_version"] == "12.3.0",
            "Invalid converter identity")
    prep = config["preprocessing"]
    require(set(prep) == {"size", "mean", "std"} and prep["size"] == 512,
            "Initial protocol uses 512 letterbox")
    for name in ("mean", "std"):
        require(len(prep[name]) == 3 and all(type(v) in (int, float) and math.isfinite(v)
                                           for v in prep[name]), "Invalid normalization")
    require(all(v > 0 for v in prep["std"]), "std must be positive")
    model, training, run = config["model"], config["training"], config["run"]
    require(model["name"] in {"unet", "segformer"}, "Unknown model")
    if model["name"] == "segformer":
        require(model["checkpoint"] == "nvidia/segformer-b0-finetuned-ade-512-512",
                "Expected ADE20K SegFormer-B0 checkpoint")
        require(bool(re.fullmatch(r"[0-9a-f]{40}", model["revision"])),
                "Pin pretrained revision to a full commit SHA")
        require(prep["mean"] == [0.485, 0.456, 0.406] and prep["std"] == [0.229, 0.224, 0.225],
                "Pinned pretrained model expects ImageNet normalization")
    else:
        require(type(model["base_channels"]) is int and model["base_channels"] >= 8
                and model["base_channels"] % 8 == 0, "Invalid U-Net width")
    require(training["optimizer"] == "adamw", "Only AdamW implemented")
    require(set(training) == {"optimizer", "lr", "weight_decay"}, "Unexpected training settings")
    for name in ("lr", "weight_decay"):
        value = training[name]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f"Invalid {name}")
    require(training["lr"] > 0, "lr must be positive")
    require(set(config["loss"]) == {"ce_weight", "dice_weight"}, "Unexpected loss settings")
    weights = list(config["loss"].values())
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in weights)
            and sum(weights) > 0, "Invalid loss weights")
    require(set(config["budget"]) == {"batch_size", "epochs", "patience"}, "Unexpected budget")
    require(all(type(v) is int and v > 0 for v in config["budget"].values()), "Invalid budget")
    require(set(run) == {"name", "seed", "color_jitter"}, "Unexpected run settings")
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", run["name"])), "Invalid run name")
    require(type(run["seed"]) is int and run["seed"] in (69420, 67, 69), "Unexpected run seed")
    require(type(run["color_jitter"]) is bool, "color_jitter must be boolean")
    require(model["name"] != "unet" or not run["color_jitter"], "U baseline has color off")
    pending = []
    if protocol["status"] != "frozen":
        pending.append("protocol not frozen")
    if not protocol["approval_reference"]:
        pending.append("instructor approval evidence missing")
    if not protocol["pilot_reference"]:
        pending.append("pilot evidence missing")
    if main and pending:
        raise ValueError("Main training blocked: " + "; ".join(pending))
    return pending
