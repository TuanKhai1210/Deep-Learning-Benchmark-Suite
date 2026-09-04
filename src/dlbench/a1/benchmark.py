"""Owner B (Khoa), reviewer C (Khải): timing protocol; not another training pipeline."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Mapping

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
    raise NotImplementedError("TODO B (Khoa): reproducible forward-only inference timing.")
