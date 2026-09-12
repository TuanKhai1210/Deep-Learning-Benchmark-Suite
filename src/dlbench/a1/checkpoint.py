"""Owner B (Khoa), reviewer C (Khải): validation-only selection, safe persistence."""

from __future__ import annotations

import math
import os
import tempfile
import torch
from collections.abc import Mapping
from pathlib import Path
from typing import Any



REQUIRED_CHECKPOINT_KEYS: frozenset[str] = frozenset({
    "model_state_dict",
    "epoch",
    "config",
    "val_metrics",
    "run_seed",
    "schema_version",
    "split_hash",
    "statistics_hash",
    "git_revision",
})

RESUME_KEYS: frozenset[str] = frozenset({
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rng_state",
})

def is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    """Compare val_macro_f1 max, val_loss min, epoch min; reject nonfinite metrics."""

    c_f1 = candidate.get("val_macro_f1", 0.0)
    c_loss = candidate.get("val_loss", float("inf"))
    c_epoch = candidate.get("epoch", float("inf"))

    if not all(math.isfinite(x) for x in [c_f1, c_loss]):
        return False

    # Case of first epoch (no incumbent yet)
    if incumbent is None:
        return True

    i_f1 = incumbent.get("val_macro_f1", 0.0)
    i_loss = incumbent.get("val_loss", float("inf"))
    i_epoch = incumbent.get("epoch", float("inf"))


    # Priority 1: Macro F1 (higher is better)
    if c_f1 > i_f1:
        return True
    if c_f1 < i_f1:
        return False

    # Priority 2: Validation loss (lower is better)
    if c_loss < i_loss:
        return True
    if c_loss > i_loss:
        return False

    # Priority 3: Epochs (lower is better)
    if c_epoch < i_epoch:
        return True

    # Default: returns False
    return False

def save_checkpoint(path: Path, payload: Mapping[str, Any], *, resume=False) -> None:
    """Atomic save; weights, epoch, config, split/stats hashes, seed, val metrics.

    Separate best.pt from last.pt. If resume is supported, last.pt also needs
    optimizer/scheduler/RNG state and documented resume granularity.
    """

    target_path = Path(path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    missing_keys = REQUIRED_CHECKPOINT_KEYS - set(payload.keys())

    if missing_keys:
        raise ValueError(f"Payload is missing required keys: {missing_keys}")

    if resume:
        missing_resume_keys = RESUME_KEYS - set(payload.keys())
        if missing_resume_keys:
            raise ValueError(f"Payload is missing keys for training resumption: {missing_resume_keys}")

    temp_file = tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f".{target_path.name}.tmp-",
        delete=False,
    )
    temp_name = temp_file.name
    temp_file.close()

    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, target_path)

    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu", resume=False) -> dict[str, Any]:
    """Load trusted checkpoint only, then verify expected keys/schema/version."""

    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError("Checkpoint does not exist.")


    try:
        payload = torch.load(target_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(target_path, map_location=map_location)


    if not isinstance(payload, dict):
        raise TypeError("Payload is not a dictionary.")

    missing_keys = REQUIRED_CHECKPOINT_KEYS - set(payload.keys())
    if missing_keys:
        raise ValueError(f"Payload is missing required keys: {missing_keys}")

    if resume:
        missing_resume_keys = RESUME_KEYS - set(payload.keys())
        if missing_resume_keys:
            raise ValueError(f"Payload is missing keys for training resumption: {missing_resume_keys}")

    return payload
