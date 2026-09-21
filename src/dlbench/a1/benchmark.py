"""Owner B (Khoa), reviewer C (Khải): timing protocol; not another training pipeline."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Mapping
import platform
import statistics
import time

if TYPE_CHECKING:
    from torch import Tensor, nn


def benchmark_inference(model: nn.Module, sample_batch: Tensor,
                        timing_config: Mapping[str, Any]) -> dict[str, Any]:
    """Same device/batch/FP32, eval + inference_mode, warm-up, repeated measures.

    Forward-only: batch already on device; exclude loader/transfer/checkpoint I/O.
    Synchronize CUDA before/after the timed region. Log device, batch, precision,
    warm-up, repetitions, median batch ms and throughput. Amortized ms/image
    at batch 128 is NOT batch-one request latency.
    """
    import torch

    for key in ("batch_size", "warmup_steps", "measurement_steps"):
        value = timing_config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if timing_config.get("scope") != "forward_only":
        raise ValueError("Only forward_only timing is supported")
    if timing_config.get("precision") != "float32":
        raise ValueError("Only float32 timing is supported")
    device = torch.device(timing_config["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA timing are supported")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if sample_batch.device != device:
        raise ValueError("Move the input batch to the timing device before measuring")
    if sample_batch.ndim == 0 or len(sample_batch) != timing_config["batch_size"]:
        raise ValueError("Input batch size does not match timing configuration")
    if sample_batch.dtype != torch.float32:
        raise ValueError("Input must be float32")
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device != device or (tensor.is_floating_point() and tensor.dtype != torch.float32):
            raise ValueError("Model must already be on the timing device in float32")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    modes = [(module, module.training) for module in model.modules()]
    samples = []
    try:
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for _ in range(timing_config["warmup_steps"]):
                model(sample_batch)
            synchronize()
            for _ in range(timing_config["measurement_steps"]):
                synchronize()
                start = time.perf_counter()
                model(sample_batch)
                synchronize()
                samples.append((time.perf_counter() - start) * 1000)
    finally:
        for module, training in modes:
            module.training = training
    median = statistics.median(samples)
    if median <= 0:
        raise ValueError("Timer resolution is insufficient")
    return {
        "scope": "forward_only", "precision": "float32",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": str(torch.__version__),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cpu_threads": torch.get_num_threads(),
        "batch_size": len(sample_batch), "input_shape": list(sample_batch.shape),
        "warmup_steps": timing_config["warmup_steps"],
        "measurement_steps": timing_config["measurement_steps"],
        "median_batch_ms": median,
        "amortized_ms_per_image": median / len(sample_batch),
        "images_per_second": len(sample_batch) * 1000 / median,
        "batch_ms_samples": samples,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "note": "Forward only; excludes loading and transfer. Amortized time is not batch-one latency.",
    }
