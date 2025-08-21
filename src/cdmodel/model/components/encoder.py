from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn


@dataclass
class EncoderState:
    history: Tensor
    h: Tensor

    # History masks for timestep-level prediction
    # The dimensions of the tensors below are:
    #
    #   batch x prediction timesteps x dialogue history timesteps
    mask: Tensor


class EncoderType(nn.Module, ABC):
    @abstractmethod
    def init(
        self, input: Tensor, lengths: Tensor, speaker_rank: Tensor
    ) -> EncoderState:
        pass


class Encoder(EncoderType):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()

        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def init(
        self, input: Tensor, lengths: Tensor, speaker_rank: Tensor
    ) -> EncoderState:
        x = nn.utils.rnn.pack_padded_sequence(
            input=input, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, h = self.rnn(x)
        history, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        mask = (speaker_rank[:, None, :-1] != speaker_rank[:, 1:].unsqueeze(2)) & (
            speaker_rank[:, None, :-1] != 0
        )

        return EncoderState(history=history, h=h, mask=mask)

    def forward(
        self, i: int, state: EncoderState, input: Tensor
    ) -> tuple[Tensor, Tensor]:
        return state.history[:, : i + 1], state.mask[:, i, : i + 1]


class EncoderCell(EncoderType):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()

        self.hidden_size: Final[int] = hidden_dim
        self.num_layers: Final[int] = num_layers

        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def init(
        self, input: Tensor, lengths: Tensor, speaker_rank: Tensor
    ) -> EncoderState:
        batch_size, num_steps, _ = input.shape

        return EncoderState(
            history=torch.zeros(
                batch_size,
                num_steps,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            ),
            h=torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=input.device,
            ),
            mask=(speaker_rank[:, None, :-1] != speaker_rank[:, 1:].unsqueeze(2))
            & (speaker_rank[:, None, :-1] != 0),
        )

    def forward(
        self, i: int, state: EncoderState, input: Tensor
    ) -> tuple[Tensor, Tensor]:
        x, state.h = self.rnn(input.unsqueeze(1), state.h)
        state.history[:, i] = x.squeeze(1)
        return state.history[:, : i + 1], state.mask[:, i, : i + 1]
