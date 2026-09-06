"""Owner B (Khoa), reviewer C (Khải): validation-only selection, safe persistence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING
import tempfile, os

if TYPE_CHECKING:
    import torch

REQUIRED_CHECKPOINT_KEYS: frozenset[str] = frozenset({
    "model_state_dict",
    "epoch",
    "config",
    "val_metrics",
    "run_seed",
})

def is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    """Compare val_macro_f1 max, val_loss min, epoch min; reject nonfinite metrics."""
    
    # Case of first epoch (no incumbent yet)
    if incumbent is None:
        return True
    
    c_f1 = candidate.get('val_macro_f1', 0.0)
    c_loss = candidate.get('val_loss', float('inf'))
    c_epoch = candidate.get('val_loss', float('inf'))
    
    i_f1 = candidate.get('val_macro_f1', 0.0)
    i_loss = candidate.get('val_loss', float('inf'))
    i_epoch = candidate.get('val_loss', float('inf'))
    
    # Nonfinite metrics handling
    if not all(math.isfinite(x) for x in [c_f1, c_loss, i_f1, i_loss]):
        return False
    
    # Due to floating point accuracy, we need eps
    eps = 1e-6
    
    # Priority 1: Macro F1 (higher is better)
    if c_f1 > i_f1 + eps:
        return True
    if c_f1 < i_f1 - eps:
        return False
    
    # Priority 2: Validation loss (lower is better)
    if c_loss < i_loss - eps:
        return True
    if c_loss > i_loss + eps:
        return False
    
    # Priority 3: Epochs (lower is better)
    if c_epoch < i_epoch:
        return True
    
    # Default: returns False
    return False

def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomic save; weights, epoch, config, split/stats hashes, seed, val metrics.

    Separate best.pt from last.pt. If resume is supported, last.pt also needs
    optimizer/scheduler/RNG state and documented resume granularity.
    """
    
    target_path = Path(path)
    
    target_path.parent.mkdir(exist_ok=True)
    
    missing_keys = REQUIRED_CHECKPOINT_KEYS - set(payload.keys())
    
    if missing_keys:
        raise ValueError(f"Payload is missing required keys: {missing_keys}")
    
    with tempfile.NamedTemporaryFile('wb', dir=target_path.parent, prefix=f".{target_path.name}.tmp-", delete=False) as temp_file:
        temp_name = temp_file.name
        
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, target_path)
            
    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load trusted checkpoint only, then verify expected keys/schema/version."""
    
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError("Checkpoint does not exist.")
    
    
    payload = torch.load(target_path, map_location=map_location)
        
    
    if not isinstance(payload, dict):
        raise TypeError("Payload is not a dictionary.")
    
    missing_keys = REQUIRED_CHECKPOINT_KEYS - set(payload.keys())
    if missing_keys:
        raise ValueError(f"Payload is missing required keys: {missing_keys}")
    
    return payload
