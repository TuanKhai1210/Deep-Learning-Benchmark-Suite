"""Owner B (Khoa), reviewer C (Khải): timing protocol; not another training pipeline."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Mapping
import time
import numpy as np

from torch import Tensor, nn
import torch


def benchmark_inference(model: nn.Module, sample_batch: Tensor,
                        timing_config: Mapping[str, Any]) -> dict[str, Any]:
    """Same device/batch/FP32, eval + inference_mode, warm-up, repeated measures.

    Forward-only: batch already on device; exclude loader/transfer/checkpoint I/O.
    Synchronize CUDA before/after the timed region. Log device, batch, precision,
    warm-up, repetitions, median batch ms and throughput. Amortized ms/image
    at batch 128 is NOT batch-one request latency.
    """
    warmup_steps = timing_config['warmup_steps']
    measurement_steps = timing_config['measurement_steps']
    
    device = sample_batch.device
    batch_size = sample_batch.size(0)
    
    precision = timing_config['precision']
    if precision == 'float32':
        model = model.to(dtype=torch.float32)
        sample_batch = sample_batch.to(dtype=torch.float32)
    
    model.eval()
    
    with torch.inference_mode():
        # Warmup step
        for _ in range(warmup_steps):
            logits = model(sample_batch)
        
        timings_ms = []
        
        # Measurement step
        for _ in range(measurement_steps):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            logits = model(sample_batch)
           
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            timings_ms.append((end_time - start_time) * 1000)
        
    median_batch_ms = float(np.median(timings_ms))
    
    throughput = batch_size / (median_batch_ms / 1000)
    
    return {
        "device": str(device),
        "precision": precision,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "median_batch_ms": median_batch_ms,
        "throughput": throughput
    }