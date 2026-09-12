"""Owner B (Khoa), reviewer C (Khải): assemble data, model, optimizer and the shared engine."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from secrets import token_hex
from typing import Any, Literal, Optional
import numpy as np
import random
import time
import torch

from dlbench.a1.contracts import EvaluationResult, FitResult
from dlbench.a1.data.loaders import build_dataloaders
from dlbench.a1.engine import train_one_epoch, evaluate_epoch
from dlbench.a1.models.registry import build_model
from dlbench.common.artifacts import (
    checkpoint_provenance,
    create_run_dir,
    save_run_metadata,
    append_history,
    save_metrics,
)
from dlbench.common.config import validate_config
from dlbench.a1.checkpoint import save_checkpoint, is_better, load_checkpoint
from dlbench.common.reproducibility import seed_everything




def generate_run_id(config: Mapping[str, Any], *, smoke: bool = False) -> str:
    """Create an A1 run ID with UTC time and a short collision-resistant suffix."""
    model_name = config.get("model", {}).get("name", "model")
    smoke_status = "smoke" if smoke else "main"
    run_seed = config["run"]["seed"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = token_hex(3)
    return f"a1_{model_name}_{smoke_status}_seed{run_seed}_{timestamp}_{suffix}"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon (M1/M2/M3/M4)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def get_optimizer(model: torch.nn.Module, training_config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = training_config.get("optimizer", "adam").lower()
    lr = float(training_config.get("learning_rate", 0.001))
    weight_decay = float(training_config.get("weight_decay", 0.0))

    supported_optimizers = ("adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam")

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = float(training_config.get("momentum", 0.9))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        momentum = float(training_config.get("momentum", 0.9))
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=float(training_config.get("alpha", 0.99)),
            eps=float(training_config.get("eps", 1e-08)),
            centered=bool(training_config.get("centered", False)),
        )
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
            lr_decay=float(training_config.get("lr_decay", 0.0)),
            weight_decay=weight_decay,
            eps=float(training_config.get("eps", 1e-10)),
        )
    elif optimizer_name == "adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=lr,
            rho=float(training_config.get("rho", 0.9)),
            eps=float(training_config.get("eps", 1e-06)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamax":
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=lr,
            betas=(float(training_config.get("beta1", 0.9)), float(training_config.get("beta2", 0.999))),
            eps=float(training_config.get("eps", 1e-08)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(
            model.parameters(),
            lr=lr,
            betas=(float(training_config.get("beta1", 0.9)), float(training_config.get("beta2", 0.999))),
            eps=float(training_config.get("eps", 1e-08)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. Supported optimizers are {', '.join(supported_optimizers)}."
        )
    return optimizer


def resolve_device(config: Mapping[str, Any] | None = None) -> torch.device:
    """Use run.device when supplied; otherwise auto-detect hardware."""
    run_config = config.get("run", {}) if config is not None else {}
    requested_device = str(run_config.get("device", "auto")).strip().lower()

    if requested_device in ("", "auto"):
        return get_device()

    try:
        return torch.device(requested_device)
    except Exception as exc:
        raise ValueError(f"Unsupported run.device: {requested_device}") from exc


def _resolve_smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Make smoke runs use explicit, small, positive budgets."""
    resolved = deepcopy(dict(config))
    budget = resolved.setdefault("budget", {})
    training = resolved.setdefault("training", {})

    try:
        max_epochs = int(budget.get("max_epochs", 0))
    except (TypeError, ValueError):
        max_epochs = 0

    if max_epochs <= 0:
        budget["max_epochs"] = 2
    else:
        budget["max_epochs"] = min(max_epochs, 2)

    try:
        batch_size = int(training.get("batch_size", 0))
    except (TypeError, ValueError):
        batch_size = 0

    if batch_size <= 0:
        training["batch_size"] = 32
    else:
        training["batch_size"] = min(batch_size, 32)

    try:
        save_frequency = int(training.get("save_frequency", 1))
    except (TypeError, ValueError):
        save_frequency = 1

    training["save_frequency"] = max(save_frequency, 1)
    return resolved


def get_scheduler(optimizer: torch.optim.Optimizer, training_config: Mapping[str, Any]) -> Any:
    scheduler_config = training_config.get("scheduler")
    if not scheduler_config:
        return None
    
    name = scheduler_config.get("name", "").lower()
    params = scheduler_config.get("parameters", {})
    
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=params.get("step_size", 10), 
            gamma=params.get("gamma", 0.1)
        )
    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=params.get("gamma", 0.9)
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=params.get("T_max", 50), 
            eta_min=params.get("eta_min", 0.0)
        )
    elif name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=params.get("mode", "min"), 
            factor=params.get("factor", 0.1), 
            patience=params.get("patience", 10),
            min_lr=params.get("min_lr", 0.0)
        )
    elif name == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer, 
            total_iters=params.get("total_iters", 100), 
            power=params.get("power", 1.0)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


def fit(config: Mapping[str, Any], *, smoke: bool = False) -> FitResult:
    """Seed -> loaders -> model -> optimizer -> train/val epochs -> artifacts.

    Main runs require strict config validation AND valid existing split/stats.
    Smoke runs: explicitly limited training/validation data, positive small
    epoch/batch budget, separate smoke directory, no official-test evaluation.
    Call engine.py, checkpoint.py and common/artifacts.py; do not duplicate them.
    Save resolved config and actual epoch/time budget. Never overwrite a run.
    """
    resolved_config = _resolve_smoke_config(config) if smoke else deepcopy(dict(config))
    validate_config(resolved_config, strict=not smoke)
    run_config = resolved_config["run"]
    # Prepare deterministic behavior
    run_seed = run_config["seed"]
    seed_everything(run_seed)

    # Create run directory
    output_root = Path(run_config.get('output_root', 'runs/a1'))
    run_id = generate_run_id(resolved_config, smoke=smoke)
    run_dir = create_run_dir(output_root, run_id)

    # Use device from config when provided; otherwise auto-detect.
    device = resolve_device(resolved_config)

    # Save run metadata and reuse the reproducibility-generated provenance when
    # the metadata payload includes the split/normalization records.
    metadata = save_run_metadata(run_dir, resolved_config)
    metadata_provenance = checkpoint_provenance(metadata)
    # Build data loaders
    dataloaders = build_dataloaders(resolved_config, smoke=smoke)

    # Build model
    model_config = resolved_config['model']
    model = build_model(model_config)
    model.to(device)

    # Setup optimizer
    training_config = resolved_config.get("training", {})
    optimizer = get_optimizer(model, training_config)
    scheduler = get_scheduler(optimizer, training_config)

    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Get training parameters
    epochs = int(resolved_config["budget"]["max_epochs"])
    if smoke:
        # For smoke tests, use minimal epochs
        epochs = min(epochs, 10)

    # Track best model
    best_val_metrics = None
    best_selection_metrics: dict[str, float | int] | None = None
    best_epoch = -1
    early_stopping_patience = int(training_config.get("early_stopping_patience", 0))
    min_delta = float(training_config.get("min_delta", 0.0))
    save_frequency = int(training_config.get("save_frequency", 1))
    early_stopping_monitor = training_config.get("early_stopping_monitor", "macro_f1")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    if min_delta < 0.0:
        raise ValueError("min_delta must be nonnegative")
    if save_frequency <= 0:
        raise ValueError("save_frequency must be positive")
    if early_stopping_monitor not in ("macro_f1", "loss"):
        raise ValueError("early_stopping_monitor must be 'macro_f1' or 'loss'")
    best_monitor_value: float | None = None
    epochs_without_improvement = 0
    best_checkpoint_path = run_dir / "best.pt"

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=dataloaders.train,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # Validate
        val_metrics: EvaluationResult = evaluate_epoch(
            model=model,
            loader=dataloaders.validation,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - start_time

        # Prepare history row
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "val_loss": val_metrics.metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_accuracy": val_metrics.metrics.accuracy,
            "train_macro_f1": train_metrics.macro_f1,
            "val_macro_f1": val_metrics.metrics.macro_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": epoch_time,
        }

        # Append to history
        append_history(run_dir, history_row)

        # Check if this is the best model so far
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "random_python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            "config": dict(resolved_config),
            "val_metrics": asdict(val_metrics.metrics),
            "run_seed": run_seed,
            "schema_version": 1,
            **metadata_provenance,
        }

        selection_metrics = {
            "val_macro_f1": val_metrics.metrics.macro_f1,
            "val_loss": val_metrics.metrics.loss,
            "epoch": epoch,
        }

        # Save resumable state at the configured frequency, at the final epoch,
        # and when early stopping cuts the run short.
        last_checkpoint_path = run_dir / "last.pt"
        should_save_last = (epoch + 1) % save_frequency == 0 or epoch == epochs - 1

        # Update best model if needed
        if is_better(selection_metrics, best_selection_metrics):
            best_val_metrics = val_metrics.metrics
            best_selection_metrics = selection_metrics
            best_epoch = epoch
            # Save best checkpoint
            save_checkpoint(best_checkpoint_path, checkpoint_payload, resume=False)

        if early_stopping_monitor == "macro_f1":
            monitor_value = val_metrics.metrics.macro_f1
            improved = best_monitor_value is None or monitor_value > best_monitor_value + min_delta
        else:
            monitor_value = val_metrics.metrics.loss
            improved = best_monitor_value is None or monitor_value < best_monitor_value - min_delta

        if improved:
            best_monitor_value = monitor_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            save_checkpoint(last_checkpoint_path, checkpoint_payload, resume=True)
            break

        if should_save_last:
            save_checkpoint(last_checkpoint_path, checkpoint_payload, resume=True)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.metrics.loss)
            else:
                scheduler.step()

    # Save final metrics
    final_metrics = {
        "eval_split": "validation",
        "val_loss": best_val_metrics.loss if best_val_metrics else 0.0,
        "val_accuracy": best_val_metrics.accuracy if best_val_metrics else 0.0,
        "val_macro_f1": best_val_metrics.macro_f1 if best_val_metrics else 0.0,
        "epoch": best_epoch,
        "timing_scope": "fit",
        "timing_units": "seconds",
    }
    save_metrics(run_dir, final_metrics)

    # Return FitResult
    history_file = run_dir / "history.csv"

    return FitResult(
        run_dir=run_dir,
        best_checkpoint=best_checkpoint_path,
        history_file=history_file,
    )


def evaluate_checkpoint(config: Mapping[str, Any], checkpoint_path: Path, *,
                        split: Literal["validation", "test"] = "validation") -> EvaluationResult:
    """Reconstruct from saved config; verify split hash, normalization and class order.

    Refuse incompatible supplied config. Evaluate the requested split with
    engine.evaluate_epoch; preserve checkpoint provenance. Test use is explicit
    and only after all model-selection decisions are fixed.
    Load only trusted checkpoints using an appropriate safe loading policy.
    """
    validate_config(dict(config), strict=True)
    if split not in ("validation", "test"):
        raise ValueError(f"Unrecognized evaluation split: {split}")

    device = resolve_device(config)
    map_location = device
    
    payload = load_checkpoint(checkpoint_path, map_location=map_location, resume=False)
    
    for section in (
        "protocol",
        "data",
        "preprocessing",
        "evaluation",
        "checkpoint",
        "model",
    ):
        if payload["config"].get(section) != config.get(section):
            raise ValueError(f"Configuration mismatch in section: {section}")
        
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        current_metadata = checkpoint_provenance(save_run_metadata(Path(temp_dir), config))

    if payload["split_hash"] != current_metadata["split_hash"]:
        raise ValueError("Checkpoint split does not match the supplied configuration.")

    if payload["statistics_hash"] != current_metadata["statistics_hash"]:
        raise ValueError("Checkpoint preprocessing does not match the supplied configuration.")
    
    model = build_model(config["model"])
    model.load_state_dict(payload["model_state_dict"])
    
    model.to(device)
    
    loaders = build_dataloaders(config, smoke=False)
    loader = loaders.validation if split == "validation" else loaders.test
    
    criterion = torch.nn.CrossEntropyLoss()
    result = evaluate_epoch(model, loader, criterion, device)
    
    return result