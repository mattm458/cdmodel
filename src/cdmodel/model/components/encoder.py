from abc import ABC, abstractmethod
from typing import Final

import torch
from torch import Tensor, nn


class EncoderType(nn.Module, ABC):
    @abstractmethod
    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        pass


class Encoder(EncoderType):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()

        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x = nn.utils.rnn.pack_padded_sequence(
            input=input, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, h = self.rnn(x)
        history, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return history, h

    def forward(
        self, i: int, history: Tensor, h: Tensor, input: Tensor
    ) -> tuple[Tensor, Tensor]:
        return history, h


class EncoderCell(EncoderType):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()

        self.hidden_size: Final[int] = hidden_dim
        self.num_layers: Final[int] = num_layers

        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def init(self, input: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, num_steps, _ = input.shape
        return torch.zeros(
            batch_size,
            num_steps - 1,
            self.hidden_size,
            device=input.device,
            dtype=input.dtype,
        ), torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=input.device,
        )

    def forward(
        self, i: int, history: Tensor, h: Tensor, input: Tensor
    ) -> tuple[Tensor, Tensor]:
        x, h = self.rnn(input, h)
        return (
            history.index_copy(
                1, torch.tensor([i], device=history.device), x.type(history.dtype)
            ),
            h,
        )
