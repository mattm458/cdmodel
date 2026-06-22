from abc import ABC, abstractmethod
from typing import Final

import torch
from torch import Tensor, nn


class EncoderType(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        learn_rnn_initial_state: bool,
    ):
        super().__init__()

        self.num_layers: Final[int] = num_layers
        self.hidden_size = hidden_dim

        if learn_rnn_initial_state:
            print("Encoder: Learning RNN initial state")
        self.h_initial = nn.Parameter(
            (
                torch.randn(num_layers, 1, hidden_dim)
                if learn_rnn_initial_state
                else torch.zeros(num_layers, 1, hidden_dim)
            ),
            requires_grad=learn_rnn_initial_state,
        )

        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    @abstractmethod
    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def forward(self, i: int, history: Tensor, h: Tensor, input: Tensor) -> Tensor:
        pass


class Encoder(EncoderType):
    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size: Final[int] = input.shape[0]

        x = nn.utils.rnn.pack_padded_sequence(
            input=input,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        x, h = self.rnn(x, self.h_initial.expand(self.num_layers, batch_size, -1))
        history, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return history, h

    def forward(self, i: int, history: Tensor, h: Tensor, input: Tensor) -> Tensor:
        return h


class EncoderCell(EncoderType):
    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, num_steps, _ = input.shape

        history = torch.zeros(
            (batch_size, num_steps, self.hidden_size),
            device=input.device,
        )

        h = self.h_initial.expand(-1, batch_size, -1)

        return history, h

    def forward(self, i: int, history: Tensor, h: Tensor, input: Tensor) -> Tensor:
        x, h = self.rnn(input, h)
        history[:, i] = x.squeeze(1)
        return h
