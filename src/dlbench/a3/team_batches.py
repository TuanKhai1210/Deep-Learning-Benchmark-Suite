"""Lock shared ABO inputs, train arbitrary assignments and assemble the final suite."""

from __future__ import annotations

import json
import re
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from .data import write_csv
from .engine import _write_json, train
from .fusion_suite import (
    RUN_FILES, TEAM_VARIANTS, _digest, _input_identity, _read_json, _unimodal_reference,
    training_record, verify_matched_heads,
)

TEAM_SEEDS = (69420, 67, 69)


def _benchmark(manifest, caches, config_path, config):
    rows, inputs = _input_identity(manifest, caches)
    package = Path(__file__).resolve().parent.parent
    source = {path.relative_to(package).as_posix(): _digest(path)
              for folder in ("a3", "common") for path in sorted((package / folder).glob("*.py"))}
    return rows, {
        "schema": 1, "config": config, "config_sha256": _digest(config_path),
        "source_sha256": source, "seeds": list(TEAM_SEEDS), "models": list(TEAM_VARIANTS), **inputs,
    }


def lock_benchmark(manifest, caches, config_path, config, lock):
    _, data = _benchmark(manifest, caches, config_path, config)
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        _check_benchmark(lock, data)
    return lock


def _check_benchmark(lock_path, data):
    if not lock_path.is_file():
        raise ValueError(f"Shared benchmark lock is missing: {lock_path}")
    current = _read_json(lock_path)
    changed = [key for key, value in data.items() if current.get(key) != value]
    if changed:
        raise ValueError("Shared benchmark differs in: " + ", ".join(changed))


def _validated_assignment(pairs):
    if (not pairs or any(model not in TEAM_VARIANTS or type(seed) is not int or seed not in TEAM_SEEDS
                         for model, seed in pairs) or len(set(pairs)) != len(pairs)):
        raise ValueError("Select distinct model/seed pairs from the five variants and seeds 69420, 67, 69")
    return sorted(pairs, key=lambda pair: (TEAM_SEEDS.index(pair[1]), TEAM_VARIANTS.index(pair[0])))


def _profile(run):
    identity = _read_json(run / "run.json")
    env = identity["environment"]
    device = identity["device"]
    gpu = env["gpus"][0]["name"] if device == "cuda" and env["gpus"] else None
    return {"device": device, "gpu": gpu, "processor": env["processor"],
            "os": env["os"], "torch": env["torch_version"], "cuda": env["torch_cuda_version"]}


def _batch_record(run, batch, benchmark, model, seed):
    return {**training_record(run, benchmark, benchmark["config"], model, seed),
            "batch_id": batch["batch_id"], "training_profile": _profile(run)}


def _batch_id(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError(f"Use a short lowercase batch ID: {name}")


def train_batch(manifest, caches, config_path, config, lock, pairs, output, device):
    selected = _validated_assignment(pairs)
    _, benchmark = _benchmark(manifest, caches, config_path, config)
    lock = Path(lock)
    _check_benchmark(lock, benchmark)
    output = Path(output)
    _batch_id(output.name)
    batch = {"schema": 1, "batch_id": output.name, "benchmark_lock_sha256": _digest(lock),
             "pairs": [[model, seed] for model, seed in selected]}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if _read_json(output / "batch.json") != batch:
            raise ValueError("This batch ID is already assigned to different runs or inputs")
    else:
        with TemporaryDirectory(dir=output.parent) as staging:
            folder = Path(staging) / "batch"
            folder.mkdir()
            _write_json(folder / "batch.json", batch)
            folder.rename(output)

    for model, seed in selected:
        name = f"{model}-{seed}"
        destination = output / name
        if destination.exists():
            _verify_batch_run(destination, batch, benchmark, model, seed)
            print(f"Already complete: {name}", flush=True)
            continue
        print(f"Training: {name}", flush=True)
        with TemporaryDirectory(dir=output) as staging:
            run = train(manifest, caches, {**config, "model": model, "seed": seed},
                        staging, name, device)
            _write_json(run / "batch-run.json", _batch_record(run, batch, benchmark, model, seed))
            run.rename(destination)
    return output


def _verify_batch_run(run, batch, benchmark, model, seed):
    record = _batch_record(run, batch, benchmark, model, seed)
    if _read_json(run / "batch-run.json") != record:
        raise ValueError(f"Training artifacts changed: {run}")
    return record


def collect_batches(manifest, caches, config_path, config, lock, batches, output):
    """Collect the 15 locked validation runs before held-out testing."""
    rows, benchmark = _benchmark(manifest, caches, config_path, config)
    lock = Path(lock)
    _check_benchmark(lock, benchmark)
    lock_sha = _digest(lock)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records = {}
    for path in map(Path, batches):
        batch = _read_json(path / "batch.json")
        _batch_id(batch["batch_id"])
        if batch["schema"] != 1 or batch["benchmark_lock_sha256"] != lock_sha:
            raise ValueError(f"Batch does not share the benchmark lock: {path}")
        for model, seed in _validated_assignment([tuple(pair) for pair in batch["pairs"]]):
            run = path / f"{model}-{seed}"
            if not run.exists():
                continue
            key = model, seed
            if key in records:
                raise ValueError(f"Duplicate completed run {run.name}: {records[key][0]} and {run}")
            records[key] = run, _verify_batch_run(run, batch, benchmark, model, seed)

    expected = [(model, seed) for seed in TEAM_SEEDS for model in TEAM_VARIANTS]
    missing = [f"{model}-{seed}" for model, seed in expected if (model, seed) not in records]
    if missing:
        raise ValueError("Missing completed runs: " + ", ".join(missing))
    runs = [records[key][1] for key in expected]
    verify_matched_heads(runs, TEAM_SEEDS)
    devices = sorted({item["training_profile"]["device"] for item in runs})
    plan = {**benchmark, "profile": "team", "distributed": True,
            "benchmark_lock_sha256": lock_sha, "training_device": "+".join(devices),
            "runs": runs, "unimodal_reference": _unimodal_reference(runs)}

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        folder = Path(staging) / "suite"
        folder.mkdir()
        write_csv(folder / "manifest.csv", tuple(rows[0]), rows)
        copy2(lock, folder / "benchmark_lock.json")
        for source, record in records.values():
            target = folder / record["directory"]
            target.mkdir()
            for name in (*RUN_FILES, "batch-run.json"):
                copy2(source / name, target / name)
            _verify_batch_run(target, record, benchmark, record["model"], record["seed"])
        _write_json(folder / "suite.json", plan)
        folder.rename(output)
    return output / "suite.json"
