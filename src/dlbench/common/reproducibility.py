"""Owner C (Khải), reviewer A (Thiên): deterministic settings and environment capture."""

from __future__ import annotations

import platform
import random
from typing import Any

import numpy as np
import torch
import torchvision


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
