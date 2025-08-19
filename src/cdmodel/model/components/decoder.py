from dataclasses import dataclass
from typing import Final, Optional

import torch
from torch import Tensor, nn

from cdmodel.model.components.attention import AdditiveAttention


@dataclass
class DecoderState:
    h: tuple[Tensor, Tensor]


class DecoderCell(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, features: list[str]
    ):
        super().__init__()

        self.hidden_size: Final[int] = hidden_size
        self.num_layers: Final[int] = num_layers

        self.attention = AttentionAdditive(hidden_dim=input_size, query_dim=96)
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )

        self.features = nn.ModuleDict()
        for f in features:
            self.features[f] = nn.Sequential(
                nn.Linear(input_size, input_size), nn.ELU(), nn.Linear(input_size, 1)
            )

    def initialize(self, batch_size: int, device) -> DecoderState:
        return DecoderState(
            h=(
                torch.zeros(
                    self.num_layers, batch_size, self.hidden_size, device=device
                ),
                torch.zeros(
                    self.num_layers, batch_size, self.hidden_size, device=device
                ),
            )
        )

    def forward(
        self, state: DecoderState, input: Tensor, mask: Optional[Tensor] = None
    ):
        batch_size = input.shape[0]
        query = state.h[0].permute(1, 0, 2).reshape(batch_size, -1)
        context, weights = self.attention(query=query, keys=input, mask=mask)
        x, state.h = self.rnn(context, state.h)

        outputs = torch.concat([self.features[f](x) for f in self.features], -1)
        return outputs, weights
