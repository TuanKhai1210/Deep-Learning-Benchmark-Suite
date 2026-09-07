"""Owner B (Khoa): MLPClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from torch import Tensor, nn

ACTIVATION_MAP = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'gelu': nn.GELU,
}

class MLPClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. Flatten internally; explain activation and regularization in the method note."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        # TODO B (Khoa): validate config and declare hidden layers from hidden_dims, activation/dropout and output head.
        # No optimizer, DataLoader, training loop or hard-coded device here.
        if "input_dim" not in parameters:
            raise ValueError("Input dimension not specified")
        if "num_classes" not in parameters:
            raise ValueError("Number of output classes not specified")
        if "hidden_dims" not in parameters:
            raise ValueError("MLP must have at least one hidden layer")
        if type(parameters["hidden_dims"]) is not list:
            raise TypeError("Hidden layers parameters must be specified as list[int]")
        if len(parameters['hidden_dims']) == 0:
            raise ValueError("MLP must have at least one hidden layer")

        input_dim = parameters['input_dim']
        num_classes = parameters['num_classes']
        hidden_dims = parameters['hidden_dims']
        if any(x <= 0 for x in hidden_dims):
            raise ValueError("Exists hidden layer with empty dimension")
        dropout_rate = parameters.get('dropout', 0.0)
        if dropout_rate < 0.0 or dropout_rate > 1.0:
            raise ValueError(f"Invalid dropout rate: {dropout_rate}")
        hidden_activation = parameters.get('hidden_activation', 'relu')
        try:
            activation_class = ACTIVATION_MAP[hidden_activation]
        except KeyError:
            raise ValueError(f"Invalid hidden activation: {hidden_activation}")

        layer_list: list[nn.Module] = [nn.Flatten(start_dim=1)] # Flattens (B, 1, 28, 28) into (B, 784)
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layer_list.append(nn.Linear(prev_dim, h_dim))
            layer_list.append(activation_class())
            if dropout_rate > 0.0:
                layer_list.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layer_list.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layer_list)


    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        Flatten internally; explain activation and regularization in the method note.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        if images.ndim < 2:
            raise ValueError("Input must include a batch dimension")
        if images.size(0) == 0:
            raise ValueError("Batch size cannot be zero")
        return self.network(images)
