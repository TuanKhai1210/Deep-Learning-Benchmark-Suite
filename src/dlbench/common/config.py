"""Owner C (Khải): Python configuration loader and an A1 readiness check.

Model files can change only model/training/run; benchmark rules live once in
protocol.py. This is not a full experiment validator: data integrity, available
hardware and correctness of implementations must be tested separately.
All dataset/artifact paths are relative to the repository working directory.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import ast
import math


class ConfigError(ValueError):
    """An invalid or unready experiment configuration."""


COMMON_SECTIONS = {
    "protocol", "data", "preprocessing", "evaluation", "checkpoint", "budget", "timing"
}
MODEL_SECTIONS = {"model", "training", "run"}
MODEL_NAMES = {"linear", "mlp", "cnn", "rnn", "transformer"}


def _read_python_config(path: Path) -> dict[str, Any]:
    """Read CONFIG literals without importing or executing configuration code.

    Allow a module docstring and one CONFIG = {...} assignment. Imports,
    function calls and computed expressions are rejected rather than executed.
    """
    if path.suffix != ".py":
        raise ConfigError(f"Expected a Python configuration (.py): {path}")
    try:
        statements = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path)).body
    except (SyntaxError, UnicodeError) as error:
        raise ConfigError(f"Invalid Python configuration {path}: {error}") from error
    if (statements and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)):
        statements = statements[1:]
    if (len(statements) != 1 or not isinstance(statements[0], ast.Assign)
            or len(statements[0].targets) != 1
            or not isinstance(statements[0].targets[0], ast.Name)
            or statements[0].targets[0].id != "CONFIG"):
        raise ConfigError(f"{path}: use one CONFIG dictionary; no imports or executable statements.")
    try:
        config = ast.literal_eval(statements[0].value)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ConfigError(f"{path}: CONFIG must contain literal data, not calls or expressions.") from error
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        raise ConfigError(f"{path}: CONFIG must be a dictionary with string keys.")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a model CONFIG and shared protocol; do not execute code or write files."""
    path = Path(path).resolve()
    model_config = _read_python_config(path)
    allowed = MODEL_SECTIONS | {"protocol_file"}
    unexpected = set(model_config) - allowed
    if unexpected:
        raise ConfigError(f"Model config may not override common sections: {sorted(unexpected)}")
    reference = model_config.get("protocol_file")
    if not isinstance(reference, str) or not reference:
        raise ConfigError("protocol_file must be a nonempty path relative to the model config file.")
    protocol_path = (path.parent / reference).resolve()
    protocol = _read_python_config(protocol_path)
    if set(protocol) != COMMON_SECTIONS:
        raise ConfigError(f"Protocol must contain exactly: {sorted(COMMON_SECTIONS)}")
    for name in COMMON_SECTIONS:
        if not isinstance(protocol[name], dict):
            raise ConfigError(f"Protocol section {name!r} must be a dictionary.")
    for name in MODEL_SECTIONS:
        if not isinstance(model_config.get(name), dict):
            raise ConfigError(f"Missing model-config dictionary: {name}")
    merged = deepcopy(protocol)
    merged.update({key: deepcopy(model_config[key]) for key in MODEL_SECTIONS})
    merged["_sources"] = {"model_config": str(path), "protocol_config": str(protocol_path)}
    return merged


def validate_config(config: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Return unresolved decisions; raise for invalid values or strict unreadiness."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ConfigError(message)

    def nonnegative_int(value: Any) -> bool:
        return type(value) is int and value >= 0

    def finite_number(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    for name in COMMON_SECTIONS | MODEL_SECTIONS:
        require(isinstance(config.get(name), dict), f"Missing config dictionary: {name}")
    protocol = config["protocol"]
    data = config["data"]
    prep = config["preprocessing"]
    evaluation = config["evaluation"]
    checkpoint = config["checkpoint"]
    budget = config["budget"]
    timing = config["timing"]
    model = config["model"]
    training = config["training"]
    run = config["run"]
    require(protocol.get("status") in ("draft", "frozen"), "protocol.status must be draft or frozen.")
    require(isinstance(protocol.get("id"), str) and bool(protocol["id"]), "protocol.id is required.")
    approved = protocol.get("approved_by")
    require(isinstance(approved, list) and all(isinstance(x, str) for x in approved),
            "approved_by must be a list of names.")
    require(data.get("dataset") == "fashion_mnist", "A1 main comparison uses fashion_mnist.")
    sizes = [data.get(key) for key in ("train_size", "validation_size", "test_size")]
    require(all(type(x) is int and x > 0 for x in sizes), "Split sizes must be positive integers.")
    require(sizes[0] + sizes[1] == 60000 and sizes[2] == 10000,
            "Train/validation must partition official 60k train; retain official 10k test.")
    require(nonnegative_int(data.get("split_seed")), "split_seed must be a nonnegative integer.")
    require(data.get("stratified") is True, "The current split implementation contract requires stratified=true.")
    for key in ("root", "split_file"):
        require(isinstance(data.get(key), str) and bool(data[key]), f"data.{key} is required.")
    require(prep.get("image_size") == [28, 28] and prep.get("channels") == 1,
            "The current A1 input contract is one-channel 28x28.")
    require(prep.get("augmentation") in ("none", "random_crop"),
            "Implement/review a new augmentation policy before adding its config value.")
    require(nonnegative_int(prep.get("crop_padding")), "crop_padding must be nonnegative.")
    require(evaluation.get("labels") == list(range(10)), "Evaluation labels must be 0..9.")
    require(evaluation.get("metrics") == ["accuracy", "macro_f1"], "Use shared accuracy/macro_f1.")
    require(evaluation.get("zero_division") == 0, "Current metric contract uses zero_division=0.")
    require(checkpoint.get("monitor") == "val_macro_f1" and checkpoint.get("mode") == "max",
            "Current checkpoint rule maximizes val_macro_f1.")
    require(checkpoint.get("tie_breaker") == ["val_loss_min", "earlier_epoch"],
            "Current tie-breakers are lower validation loss then earlier epoch.")
    require(isinstance(model.get("name"), str) and model["name"] in MODEL_NAMES, "Unknown model.name.")
    require(isinstance(model.get("parameters"), dict), "model.parameters must be a dictionary.")
    require(nonnegative_int(run.get("seed")), "run.seed must be a nonnegative integer.")
    seeds = budget.get("run_seeds")
    require(isinstance(seeds, list) and bool(seeds)
            and all(nonnegative_int(x) for x in seeds), "budget.run_seeds must list seeds.")
    require(len(seeds) == len(set(seeds)), "budget.run_seeds must not contain duplicates.")
    require(run["seed"] in seeds, "run.seed must belong to the shared run_seeds.")
    for key in ("device", "output_root"):
        require(isinstance(run.get(key), str) and bool(run[key]), f"run.{key} is required.")
    for table, key in ((budget, "max_epochs"), (budget, "tuning_trials_per_model"),
                       (training, "batch_size"), (training, "early_stopping_patience"),
                       (timing, "batch_size"), (timing, "warmup_steps"),
                       (timing, "measurement_steps")):
        require(nonnegative_int(table.get(key)), f"{key} must be a nonnegative integer.")
    supported_optimizers = {"adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"}
    require(training.get("optimizer") in supported_optimizers,
            f"Unsupported optimizer: {training.get('optimizer')}. Supported optimizers are {', '.join(sorted(supported_optimizers))}.")
    require(finite_number(training.get("learning_rate")) and training["learning_rate"] > 0,
            "learning_rate must be finite and positive.")
    require(finite_number(training.get("weight_decay")) and training["weight_decay"] >= 0,
            "weight_decay must be finite and nonnegative.")
    
    scheduler = training.get("scheduler")
    if scheduler is not None:
        require(isinstance(scheduler, dict), "scheduler must be a dictionary if provided.")
        require(isinstance(scheduler.get("name"), str), "scheduler.name must be a string.")
        require(isinstance(scheduler.get("parameters", {}), dict), "scheduler.parameters must be a dictionary.")
        supported_schedulers = {"step", "exponential", "cosine", "reduce_on_plateau", "polynomial"}
        require(scheduler["name"].lower() in supported_schedulers,
                f"Unsupported scheduler: {scheduler['name']}. Supported schedulers are {', '.join(sorted(supported_schedulers))}.")
    require(timing.get("precision") == "float32", "Current timing protocol uses float32.")
    require(timing.get("scope") == "forward_only", "Current timing protocol is forward_only.")
    require(isinstance(timing.get("device"), str) and bool(timing["device"]), "timing.device is required.")

    unresolved: list[str] = []
    if protocol["status"] != "frozen":
        unresolved.append("protocol.status is draft; review and freeze the main protocol.")
    if len({name.strip() for name in approved if name.strip()}) < 3:
        unresolved.append("protocol.approved_by must record all three distinct reviewers.")
    mean, std = prep.get("mean"), prep.get("std")
    require(isinstance(mean, list) and isinstance(std, list), "mean/std must be lists.")
    if not mean and not std:
        unresolved.append("preprocessing.mean/std are unmeasured; compute from train only.")
    else:
        require(len(mean) == len(std) == prep["channels"], "mean/std length must equal channels.")
        require(all(finite_number(x) for x in mean + std)
                and all(x > 0 for x in std), "mean/std must be finite; std must be positive.")
    for label, value in (
        ("budget.max_epochs", budget["max_epochs"]),
        ("budget.tuning_trials_per_model", budget["tuning_trials_per_model"]),
        ("training.batch_size", training["batch_size"]),
        ("timing.batch_size", timing["batch_size"]),
        ("timing.warmup_steps", timing["warmup_steps"]),
        ("timing.measurement_steps", timing["measurement_steps"]),
    ):
        if value == 0:
            unresolved.append(f"{label} is undecided (0).")
    if timing["device"] == "TBD":
        unresolved.append("timing.device must identify the agreed benchmark device.")
    if strict and unresolved:
        raise ConfigError("Not ready for main experiments:\n- " + "\n- ".join(unresolved))
    return unresolved
