"""Owner C (Khải): working, lazy name-to-model routing. Constructors are still TODO."""

from __future__ import annotations

from importlib import import_module
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

MODEL_REGISTRY = {
    "linear": ("dlbench.a1.models.linear", "LinearClassifier"),
    "mlp": ("dlbench.a1.models.mlp", "MLPClassifier"),
    "cnn": ("dlbench.a1.models.cnn", "CNNClassifier"),
    "rnn": ("dlbench.a1.models.rnn", "RNNClassifier"),
    "transformer": ("dlbench.a1.models.transformer", "TransformerClassifier"),
}


def build_model(model_config: Mapping[str, Any]) -> nn.Module:
    """No torch import until called; pass only the model table, not full config."""
    name = model_config.get("name")
    if not isinstance(name, str) or name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; expected {sorted(MODEL_REGISTRY)}.")
    module_name, class_name = MODEL_REGISTRY[name]
    constructor = getattr(import_module(module_name), class_name)
    return constructor(model_config["parameters"])
