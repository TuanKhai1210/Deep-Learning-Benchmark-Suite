"""Owner B (Khoa): RNNClassifier. Same public shape contract as every other model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import torch
from torch import Tensor, nn


class RNNClassifier(nn.Module):
    """Implement from scratch using PyTorch layers. For rows: [B,1,28,28] -> [B,28,28], batch_first=True. Document timestep and hidden-state selection."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        super().__init__()
        if "input_size" not in parameters:
            raise ValueError("Input size not specified")
        if "num_classes" not in parameters:
            raise ValueError("Number of output classes not specified")
        if "cell" not in parameters:
            raise ValueError("RNN cell type not specified")
        if parameters['cell'] not in ('lstm', 'gru'):
            raise ValueError(f"Invalid RNN cell type: {parameters['cell']}. Cell type must be 'lstm' or 'gru'")
        if "representation" not in parameters:
            raise ValueError("Sequence representation not specified")
        if parameters['representation'] not in ('rows', 'columns'):
            raise ValueError(f"Invalid image representation: {parameters['representation']}. Representation must be 'rows' or 'columns'")
        if "hidden_size" not in parameters:
            raise ValueError("Hidden size not specified")

        input_size = parameters['input_size']
        num_classes = parameters['num_classes']
        self.cell = parameters['cell']
        self.representation = parameters['representation']
        hidden_size = parameters['hidden_size']
        num_layers = parameters.get('num_layers', 1)
        self.bidirectional = parameters.get('bidirectional', False)
        num_directions = 2 if self.bidirectional else 1
        dropout = parameters.get('dropout', 0.0)
        
        RNNClass = nn.LSTM if self.cell == 'lstm' else nn.GRU
        self.rnn = RNNClass(input_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        self.fc = nn.Linear(num_directions * hidden_size, num_classes)
        
    def forward(self, images: Tensor) -> Tensor:
        """Input float32 [B,1,28,28]; output raw logits [B,10].

        For rows: [B,1,28,28] -> [B,28,28], batch_first=True. Document timestep and hidden-state selection.
        Acceptance: batch sizes 1 and 7, finite outputs, backward updates weights.
        """
        x = images.squeeze(1)
        if self.representation == 'columns':
            x = x.permute(0, 2, 1)
        
        # LSTM outputs: output, (hn, cn) where
        # - output (B, seq_len, directions * hidden_size): output features h_t of the LAST layer
        # - hn (directions * num_layers, B, hidden_size): final hidden state for each layer
        # - cn (directionss * num_layers, B, hidden_size): final cell state for each layer
        # if bidirectioal=True, output contains a concatenation of the forward and reverse hidden states
        # GRU outputs: output, hn
        output, hidden = self.rnn(x)
        h_n = hidden[0] if self.cell == 'lstm' else hidden
        if self.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final = h_n[-1]
        return self.fc(final)