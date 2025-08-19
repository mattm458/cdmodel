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
        self,
        input_dim: int,
        additional_decoder_dim: int,
        hidden_dim: int,
        num_layers: int,
        features: list[str],
    ):
        super().__init__()

        self.hidden_size: Final[int] = hidden_dim
        self.num_layers: Final[int] = num_layers

        self.attention = AdditiveAttention(
            hidden_dim=input_dim, query_dim=hidden_dim * num_layers
        )
        self.rnn = nn.LSTM(
            input_dim + additional_decoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, len(features)),
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
        self,
        state: DecoderState,
        input: Tensor,
        additional_decoder_in: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        batch_size = input.shape[0]
        query = state.h[0].permute(1, 0, 2).reshape(batch_size, -1)
        context, weights = self.attention(query=query, keys=input, mask=mask)

        if additional_decoder_in is not None:
            context = torch.concat([context, additional_decoder_in], dim=-1)

        x, state.h = self.rnn(context, state.h)

        outputs = self.linear(x)
        return outputs, weights
