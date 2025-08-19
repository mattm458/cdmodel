from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn


@dataclass
class EncoderState:
    history: Tensor
    h: tuple[Tensor, Tensor]


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )

    def initialize(self, input: Tensor, lengths: Tensor):
        x = nn.utils.rnn.pack_padded_sequence(
            input=input, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, h = self.rnn(x)
        history, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return EncoderState(history=history, h=h)

    def forward(self, i: int, state: EncoderState, input: Tensor) -> Tensor:
        return state.history[:, : i + 1]


class EncoderCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()

        self.hidden_size: Final[int] = hidden_size
        self.num_layers: Final[int] = num_layers

        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )

    def initialize(self, input: Tensor, lengths: Tensor):
        batch_size, num_steps, _ = input.shape

        return EncoderState(
            history=torch.zeros(
                batch_size,
                num_steps,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            ),
            h=(
                torch.zeros(
                    self.num_layers,
                    batch_size,
                    self.hidden_size,
                    device=input.device,
                ),
                torch.zeros(
                    self.num_layers,
                    batch_size,
                    self.hidden_size,
                    device=input.device,
                ),
            ),
        )

    def forward(self, i: int, state: EncoderState, input: Tensor) -> Tensor:
        x, state.h = self.rnn(input.unsqueeze(1), state.h)
        state.history[:, i] = x.squeeze(1)
        return state.history[:, : i + 1]
