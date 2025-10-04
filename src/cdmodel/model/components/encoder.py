from abc import ABC, abstractmethod
from typing import Final

import torch
from torch import Tensor, nn


class EncoderType(nn.Module, ABC):
    @abstractmethod
    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def forward(self, i: int, history: Tensor, h: Tensor, input: Tensor) -> Tensor:
        pass


class Encoder(EncoderType):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        learn_rnn_initial_state: bool,
    ):
        super().__init__()

        self.h_initial: nn.Parameter | None = None
        if learn_rnn_initial_state:
            print("Encoder: Learning RNN initial state")
            self.h_initial = nn.Parameter(torch.randn(num_layers, hidden_dim))

        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = input.shape[0]

        x = nn.utils.rnn.pack_padded_sequence(
            input=input[:, :-1],
            lengths=lengths.cpu() - 1,
            batch_first=True,
            enforce_sorted=False,
        )
        if self.h_initial is not None:
            x, h = self.rnn(x, self.h_initial.unsqueeze(1).expand(-1, batch_size, -1))
        else:
            x, h = self.rnn(x)

        history, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return history, h

    def forward(self, i: int, history: Tensor, h: Tensor, input: Tensor) -> Tensor:
        return h


class EncoderCell(EncoderType):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        learn_rnn_initial_state: bool,
    ):
        super().__init__()

        self.hidden_size: Final[int] = hidden_dim
        self.num_layers: Final[int] = num_layers

        self.h_initial: nn.Parameter | None = None
        if learn_rnn_initial_state:
            print("EncoderCell: Learning RNN initial state")
            self.h_initial = nn.Parameter(torch.randn(num_layers, hidden_dim))

        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, num_steps, _ = input.shape

        history = torch.zeros(
            batch_size,
            num_steps - 1,
            self.hidden_size,
            device=input.device,
        )

        h = (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
            )
            if self.h_initial is None
            else self.h_initial.unsqueeze(1).expand(-1, batch_size, -1)
        )

        return history, h

    def forward(self, i: int, history: Tensor, h: Tensor, input: Tensor) -> Tensor:
        x, h = self.rnn(input, h)
        history[:, i] = x.squeeze(1)
        return h
