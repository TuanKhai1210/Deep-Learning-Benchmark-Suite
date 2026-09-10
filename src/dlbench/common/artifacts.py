"""Owner C (Khải): run directories and provenance; never overwrite an existing run."""
from __future__ import annotations
import hashlib
import json
import math
import re
import subprocess

from dlbench.common.reproducibility import collect_environment
from collections.abc import Mapping
from pathlib import Path
from typing import Any

def _encode_json(payload: Any) -> str:
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _write_text_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as file:
        file.write(text)


def _statistics_digest(values: Mapping[str, Any]) -> str:
    text = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_normalization_record(
    preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    mean = preprocessing.get("mean")
    std = preprocessing.get("std")

    if not isinstance(mean, list) or not isinstance(std, list):
        raise TypeError("mean and std must be lists")

    if len(mean) != 1 or len(std) != 1:
        raise ValueError(
            "Expected one measured mean and std for grayscale input"
        )

    for name, values in (("mean", mean), ("std", std)):
        value = values[0]

        if type(value) not in (int, float):
            raise TypeError(f"{name} must contain a number")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if std[0] <= 0:
        raise ValueError("std must be positive")

    statistics = {
        "mean": mean,
        "std": std,
    }

    return {
        **statistics,
        "sha256": _statistics_digest(statistics),
    }


def _build_split_record(
    split_path: Path,
    recorded_path: str,
) -> dict[str, str]:
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split manifest does not exist: {split_path}"
        )

    return {
        "file": recorded_path,
        "sha256": hashlib.sha256(
            split_path.read_bytes()
        ).hexdigest(),
    }

def create_run_dir(output_root: Path, run_id: str) -> Path:
    """Create a new run directory with exist_ok=False; reject unsafe run IDs."""
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")

    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run_id may contain only ASCII letters, digits, underscores, and hyphens")

    reserved_names = {"CON", "PRN", "AUX", "NUL"}
    reserved_names.update(f"COM{i}" for i in range(1, 10))
    reserved_names.update(f"LPT{i}" for i in range(1, 10))

    if run_id.upper() in reserved_names:
        raise ValueError("run_id is a reserved Windows name")

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir: Path, config: Mapping[str, Any]) -> None:
    sources = config.get("_sources")
    if not isinstance(sources, Mapping):
        raise TypeError("config['_sources'] must be a mapping")

    source_payloads: dict[str, bytes] = {}
    source_records: dict[str, dict[str, str]] = {}

    for name, location in sources.items():
        source_path = Path(location)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Config source does not exist: {source_path}"
            )

        content = source_path.read_bytes()
        snapshot_name = f"{name}.py"

        source_payloads[snapshot_name] = content
        source_records[name] = {
            "snapshot": f"sources/{snapshot_name}",
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    split_location = config["data"]["split_file"]
    split_record = _build_split_record(
        Path(split_location),
        str(split_location),
    )
    normalization_record = _build_normalization_record(
        config["preprocessing"]
    )

    repository_root = Path(__file__).resolve().parents[3]

    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    git_dirty = bool(git_status.strip())
    environment = collect_environment()

    metadata = {
        "schema_version": 1,
        "protocol_id": config["protocol"]["id"],
        "model_name": config["model"]["name"],
        "run_seed": config["run"]["seed"],
        "run_mode": config["run"]["mode"],
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "code_state": (
            "dirty_non_release" if git_dirty else "clean"
        ),
        "split": split_record,
        "normalization": normalization_record,
        "sources": source_records,
    }

    texts = {
        "config.json": _encode_json(config),
        "environment.json": _encode_json(environment),
        "metadata.json": _encode_json(metadata),
    }

    sources_dir = run_dir / "sources"
    sources_dir.mkdir(exist_ok=False)

    for snapshot_name, content in source_payloads.items():
        destination = sources_dir / snapshot_name
        with destination.open("xb") as file:
            file.write(content)

    for filename, text in texts.items():
        _write_text_new(run_dir / filename, text)


def append_history(run_dir: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch; stable CSV schema, explicit units, no fabricated values.

    Columns: epoch, train_loss, val_loss, train_accuracy, val_accuracy,
    train_macro_f1, val_macro_f1, learning_rate, epoch_seconds.
    """
    raise NotImplementedError("TODO C (Khải): write/validate one CSV history row.")


def save_metrics(run_dir: Path, metrics: Mapping[str, Any]) -> None:
    """Save metrics.json: evaluation split, scores, epoch, timing scope/units."""
    raise NotImplementedError("TODO C (Khải): serialize real metrics with provenance.")
