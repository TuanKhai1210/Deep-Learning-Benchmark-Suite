"""Owner C (Khải): run directories and provenance; never overwrite an existing run."""
from __future__ import annotations
import hashlib
import csv
import io
import json
import math
import re
import subprocess
import os
import tempfile
from uuid import uuid4
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dlbench.common.reproducibility import collect_environment


CHECKPOINT_PROVENANCE_KEYS: frozenset[str] = frozenset({
    "schema_version",
    "split_hash",
    "statistics_hash",
    "git_revision",
})

HISTORY_COLUMNS = (
    "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
    "train_macro_f1", "val_macro_f1", "learning_rate", "epoch_seconds",
)


def _data_records(config: Mapping[str, Any]) -> tuple[dict, dict]:
    location = config["data"]["split_file"]
    return (
        _build_split_record(Path(location), str(location)),
        _build_normalization_record(config["preprocessing"]),
    )


def compute_data_provenance(config: Mapping[str, Any]) -> dict[str, str]:
    """Hash current split bytes and normalization; no files or RNG are changed.

    Requires only data.split_file and preprocessing.mean/std. Does not validate
    split membership or compare model/config compatibility for the caller.
    """
    split, normalization = _data_records(config)
    return {"split_hash": split["sha256"],
            "statistics_hash": normalization["sha256"]}


def _validate_number(name: str, value: Any, *, unit_interval: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (unit_interval and value > 1):
        raise ValueError(f"{name} is outside its allowed range")


def _validate_history_row(row: Mapping[str, Any]) -> None:
    if set(row) != set(HISTORY_COLUMNS):
        raise ValueError("History row must match HISTORY_COLUMNS exactly")
    if type(row["epoch"]) is not int or row["epoch"] < 0:
        raise ValueError("epoch must be a nonnegative integer")
    for key in HISTORY_COLUMNS[1:]:
        _validate_number(key, row[key], unit_interval=(
            key.endswith("accuracy") or key.endswith("macro_f1")))


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
        raise ValueError(
            "run_id may contain only ASCII letters, digits, "
            "underscores, and hyphens"
        )

    reserved_names = {"CON", "PRN", "AUX", "NUL"}
    reserved_names.update(f"COM{i}" for i in range(1, 10))
    reserved_names.update(f"LPT{i}" for i in range(1, 10))

    if run_id.upper() in reserved_names:
        raise ValueError("run_id is a reserved Windows name")

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def checkpoint_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Map run metadata to the exact provenance keys required by checkpoints."""
    split = metadata.get("split")
    normalization = metadata.get("normalization")

    if not isinstance(split, Mapping):
        raise KeyError("Missing metadata field: split")
    if not isinstance(normalization, Mapping):
        raise KeyError("Missing metadata field: normalization")

    field_paths = {
        "schema_version": (metadata, "schema_version"),
        "split_hash": (split, "sha256"),
        "statistics_hash": (normalization, "sha256"),
        "git_revision": (metadata, "git_revision"),
    }
    provenance: dict[str, Any] = {}

    for output_key, (record, source_key) in field_paths.items():
        if source_key not in record:
            raise KeyError(f"Missing metadata field for {output_key}")
        provenance[output_key] = record[source_key]

    return provenance


def save_run_metadata(
    run_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist run provenance and return the exact metadata written to disk."""
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
    split_record, normalization_record = _data_records(config)

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

    return metadata


def append_history(run_dir: Path, row: Mapping[str, Any]) -> None:
    """Append one epoch; stable CSV schema, explicit units, no fabricated values.

    Columns: epoch, train_loss, val_loss, train_accuracy, val_accuracy,
    train_macro_f1, val_macro_f1, learning_rate, epoch_seconds.
    """
    _validate_history_row(row)
    path = run_dir / "history.csv"
    exists = path.exists()
    last_epoch = -1
    if exists:
        content = path.read_text(encoding="utf-8")
        if not content.endswith("\n"):
            raise ValueError("History is empty or has an incomplete final line")
        reader = csv.DictReader(io.StringIO(content, newline=""))
        if reader.fieldnames != list(HISTORY_COLUMNS):
            raise ValueError("Existing history header does not match HISTORY_COLUMNS")
        for saved in reader:
            try:
                parsed = {key: int(saved[key]) if key == "epoch" else float(saved[key])
                          for key in HISTORY_COLUMNS}
                if set(saved) != set(HISTORY_COLUMNS):
                    raise ValueError("Unexpected CSV fields")
                _validate_history_row(parsed)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError("Existing history contains an invalid row") from exc
            if parsed["epoch"] <= last_epoch:
                raise ValueError("Existing history epochs must increase strictly")
            last_epoch = parsed["epoch"]
    if row["epoch"] <= last_epoch:
        raise ValueError("Epoch must be greater than the last saved epoch")
    # One writer per run. Validate before opening so rejected rows preserve bytes.
    with path.open("a" if exists else "x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_metrics(
    run_dir: Path, metrics: Mapping[str, Any], *, overwrite: bool = False,
) -> None:
    """Save split-specific metrics; explicit overwrite backs up old bytes first.

    A single writer owns each run. Replacement uses a temporary file on the
    same filesystem; validation or backup failure leaves the old file intact.
    """
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    required = {"eval_split", "epoch", "timing_scope", "timing_units"}
    if not required <= set(metrics):
        raise ValueError(f"Missing metrics fields: {sorted(required - set(metrics))}")
    split = metrics["eval_split"]
    if split not in ("validation", "test"):
        raise ValueError("eval_split must be validation or test")
    if type(metrics["epoch"]) is not int or metrics["epoch"] < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if not isinstance(metrics["timing_scope"], str) or not metrics["timing_scope"].strip():
        raise ValueError("timing_scope must be a nonempty string")
    if metrics["timing_units"] != "seconds":
        raise ValueError("timing_units must be seconds")
    prefix = "val" if split == "validation" else "test"
    for suffix in ("loss", "accuracy", "macro_f1"):
        key = f"{prefix}_{suffix}"
        if key not in metrics:
            raise ValueError(f"Missing metrics field: {key}")
        _validate_number(key, metrics[key], unit_interval=suffix != "loss")
    opposite = "test" if prefix == "val" else "val"
    if any(f"{opposite}_{suffix}" in metrics for suffix in ("loss", "accuracy", "macro_f1")):
        raise ValueError("Metric names conflict with eval_split")
    # Serialize before opening. Validation and test must not overwrite each other.
    text = _encode_json(dict(metrics))
    filename = "metrics.json" if split == "validation" else "metrics_test.json"
    target = run_dir / filename
    if not overwrite or not target.exists():
        _write_text_new(target, text)
        return

    old_bytes = target.read_bytes()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=run_dir,
            prefix=f".{filename}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        backups = run_dir / "backups"
        backups.mkdir(exist_ok=True)
        backup = backups / f"{target.stem}-{uuid4().hex}.json"
        with backup.open("xb") as file:
            file.write(old_bytes)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
