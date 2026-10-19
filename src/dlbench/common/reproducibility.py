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


def _get_system_ram_bytes() -> int | None:
    """Return total physical RAM in bytes, or None if unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except (ImportError, AttributeError):
        pass
    # Fallback: Windows WMI / Linux meminfo
    try:
        if platform.system() == "Windows":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError, AttributeError):
        pass
    return None


def collect_environment() -> dict[str, Any]:
    """Return Python/dependency versions, hardware, CUDA and determinism settings."""
    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": props.total_mem,
                "total_memory_gb": round(props.total_mem / (1024 ** 3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    # -- Training precision and TF32 flags ----------------------------------
    precision_info: dict[str, Any] = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if cuda_available:
        precision_info["cuda_matmul_allow_tf32"] = (
            torch.backends.cuda.matmul.allow_tf32
        )
    else:
        precision_info["cuda_matmul_allow_tf32"] = None

    # -- cuDNN information --------------------------------------------------
    cudnn_info: dict[str, Any] = {
        "available": torch.backends.cudnn.is_available(),
        "enabled": torch.backends.cudnn.enabled,
    }
    if torch.backends.cudnn.is_available():
        cudnn_info["version"] = torch.backends.cudnn.version()
        cudnn_info["allow_tf32"] = torch.backends.cudnn.allow_tf32
        cudnn_info["benchmark"] = torch.backends.cudnn.benchmark
        cudnn_info["deterministic"] = torch.backends.cudnn.deterministic
    else:
        cudnn_info["version"] = None
        cudnn_info["allow_tf32"] = None
        cudnn_info["benchmark"] = None
        cudnn_info["deterministic"] = None

    # -- System RAM ---------------------------------------------------------
    ram_bytes = _get_system_ram_bytes()
    ram_info: dict[str, Any] = {
        "total_bytes": ram_bytes,
        "total_gb": round(ram_bytes / (1024 ** 3), 2) if ram_bytes else None,
    }

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
        "system_ram": ram_info,
        "precision": precision_info,
        "cudnn": cudnn_info,
    }
