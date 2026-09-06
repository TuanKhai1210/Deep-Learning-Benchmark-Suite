"""Owner B (Khoa), reviewer C (Khải): one-epoch mechanics only, shared by every model.

trainer.py owns multi-epoch orchestration, validation checkpoint decisions and
artifacts. Do not add a second training loop inside any model module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import torch
from torch import nn

from dlbench.a1.contracts import EpochMetrics, EvaluationResult, Predictions, Batch
from dlbench.a1.metrics import classification_metrics

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> EpochMetrics:
    """Train mode, move batch, zero_grad, logits, CE, backward, optimizer step.

    Aggregate loss by sample count; compute metrics across epoch predictions.
    Do not apply softmax before CrossEntropyLoss; do not read the test loader.
    """
    model.train()
    total_loss: float = 0.0
    total_samples: int = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    batch: Batch
    for batch in loader:
        inputs, targets = batch['images'], batch['labels']

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(preds.detach().cpu().tolist())

    metrics = classification_metrics(all_targets, all_predictions)
    average_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return EpochMetrics(
        loss=average_loss,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        num_samples=total_samples,
    )


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> EvaluationResult:
    """eval + inference_mode; no optimizer update; retain IDs and predictions.

    Aggregate loss by sample count, labels fixed 0..9, macro-F1 over the split.
    Compute probabilities only for outputs/analysis, not as input to CE.
    """
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_sample_ids: list[str] = []
    all_targets: list[int] = []
    all_predicted_labels: list[int] = []
    all_probabilities: list[list[float]] = []

    sample_counter = 0

    with torch.inference_mode():
        batch: Batch
        for batch in loader:
            inputs, labels, batch_ids = batch['images'], batch['labels'], batch.get("sample_ids")

            batch_size = labels.size(0)

            if batch_ids is None:
                raise ValueError("sample_ids missing from Batch.")
            if len(batch_ids) != batch_size:
                raise ValueError(f"sample_ids length ({len(batch_ids)} does not match batch size ({batch_size}))")

            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_ids is not None:
                all_sample_ids.extend([str(i) for i in batch_ids])
            else:
                all_sample_ids.extend([f"sample_{i}" for i in range(sample_counter, sample_counter + batch_size)])
            sample_counter += batch_size

            all_targets.extend(labels.detach().cpu().tolist())
            all_predicted_labels.extend(preds.detach().cpu().tolist())
            all_probabilities.extend(probs.detach().cpu().tolist())

    metrics = classification_metrics(all_targets, all_predicted_labels)
    average_loss = running_loss / total_samples if total_samples > 0 else 0.0

    epoch_metrics = EpochMetrics(
        loss=average_loss,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        num_samples=total_samples,
    )

    predictions = Predictions(
        sample_ids=all_sample_ids,
        targets=all_targets,
        predicted_labels=all_predicted_labels,
        probabilities=all_probabilities,
    )

    return EvaluationResult(metrics=epoch_metrics, predictions=predictions)