"""Owner C (Khải), reviewer A (Thiên): deterministic settings and environment capture."""

from __future__ import annotations

import platform
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torchvision


RNG_STATE_KEYS: frozenset[str] = frozenset({
    "python",
    "numpy",
    "torch_cpu",
    "torch_cuda",
})


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch; configure and log determinism explicitly.

    This does not guarantee bitwise equivalence across devices/library versions.
    Keep the DataLoader generator separate from model initialization RNG.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """Top-level, picklable worker_init_fn (Windows multiprocessing compatible)."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU and all available CUDA RNG states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a complete RNG snapshot captured by :func:`capture_rng_state`."""
    if not isinstance(state, Mapping):
        raise TypeError("RNG state must be a mapping")

    missing_keys = RNG_STATE_KEYS - set(state)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"RNG state is missing required keys: {missing}")

    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise TypeError("torch_cuda RNG state must be a list or tuple")

    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Cannot restore CUDA RNG state because CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA RNG state count does not match the available device count"
            )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def collect_environment() -> dict[str, Any]:
    """Return Python/dependency versions, hardware, CUDA and determinism settings."""
    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            gpus.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
            })
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torchvision_version": str(torchvision.__version__),
        "gpus": gpus,
    }
