"""Shared FP32 segmentation loop. No classification engine assumptions."""
import torch
from .metrics import SegmentationMetrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    previous_mode = model.training
    model.train(training)
    metrics = SegmentationMetrics()
    weighted_loss, valid_pixels, batches = 0.0, 0, 0
    try:
        with torch.set_grad_enabled(training):
            for batch in loader:
                image = batch["image"].to(device)
                target = batch["mask"].to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                loss = criterion(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite segmentation loss")
                count = int((target != 255).sum())
                if training and count:
                    loss.backward()
                    optimizer.step()
                metrics.update(logits.argmax(1), target)
                weighted_loss += float(loss.detach()) * count
                valid_pixels += count
                batches += 1
        if not batches or not valid_pixels:
            raise ValueError("Epoch has no supervised pixels")
        result = metrics.compute()
        result["loss"] = weighted_loss / valid_pixels
        result["batches"] = batches
        return result
    finally:
        model.train(previous_mode)
