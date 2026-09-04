"""Shared interfaces. Changes require review by all module owners.

Owner C (Khải), reviewers A/B (Thiên/Khoa). Annotations do not enforce tensor shapes at runtime;
implement and enable the corresponding contract tests before main runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader


class Batch(TypedDict):
    images: Tensor  # float32 [B, 1, 28, 28], normalized with training-only stats
    labels: Tensor  # int64 [B], values 0..9
    sample_ids: list[str]  # e.g. fashion_mnist:official_train:123


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float  # sample-weighted mean
    accuracy: float  # 0..1, computed across the entire split
    macro_f1: float  # 0..1, not the mean of per-batch F1 values
    num_samples: int


@dataclass(frozen=True)
class Predictions:
    sample_ids: list[str]
    targets: list[int]
    predicted_labels: list[int]
    probabilities: list[list[float]]  # [N, 10]; softmax for analysis, NOT CE input


@dataclass(frozen=True)
class EvaluationResult:
    metrics: EpochMetrics
    predictions: Predictions


@dataclass(frozen=True)
class FitResult:
    run_dir: Path
    best_checkpoint: Path
    history_file: Path


@dataclass(frozen=True)
class SplitManifest:
    dataset: str
    split_seed: int
    train_indices: list[int]  # indices in official training set
    validation_indices: list[int]  # indices in official training set
    test_indices: list[int]  # indices in official test set: a separate namespace
