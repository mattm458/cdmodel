from typing import Final, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.model.components.components import Attention, AttentionActivation


class ISTOneShotEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_features: int,
        att_dim: int,
        att_activation: AttentionActivation,
    ):
        super().__init__()

        self.encoder = nn.GRU(
            num_features, token_dim // 2, bidirectional=True, batch_first=True
        )

        self.attention = Attention(
            history_in_dim=token_dim,
            context_dim=att_dim,
            att_dim=att_dim,
            scoring_activation=att_activation,
        )

    def forward(
        self, features: Tensor, features_len: Tensor, tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch_size: Final[int] = features.shape[0]

        _, h = self.encoder(
            nn.utils.rnn.pack_padded_sequence(
                features, features_len, batch_first=True, enforce_sorted=False
            )
        )

        embeddings, weights = self.attention(
            history=F.tanh(tokens).repeat((batch_size, 1, 1)),
            context=torch.cat([h[0], h[1]], dim=-1),
        )

        return embeddings.squeeze(1), weights.squeeze(1)


class ISTIncrementalEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_features: int,
        att_dim: int,
        att_activation: AttentionActivation,
        accumulator_decay: float = 1.0,
    ):
        super().__init__()

        self.accumulator_decay: Final[float] = 1.0 - accumulator_decay

        self.token_dim: Final[int] = token_dim

        self.encoder = nn.GRUCell(num_features, token_dim)

        self.attention = Attention(
            history_in_dim=token_dim,
            context_dim=att_dim,
            att_dim=att_dim,
            scoring_activation=att_activation,
        )

    def get_hidden(self, batch_size: int, device, precision: str) -> Tensor:
        dtype = torch.half if precision == "16-true" else torch.float
        return torch.zeros((batch_size, self.token_dim), device=device, dtype=dtype)

    def get_accumulator(self, batch_size: int, device, precision: str) -> Tensor:
        dtype = torch.half if precision == "16-true" else torch.float
        return torch.zeros((batch_size, self.token_dim), device=device, dtype=dtype)

    def forward(
        self,
        features: Tensor,
        h: Tensor,
        accumulator: Tensor,
        mask: Tensor,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h = h.clone()

        h_new = self.encoder(features[mask], h[mask])
        h[mask] = h_new

        embeddings, weights = self.attention(
            history=F.tanh(tokens).repeat((h_new.shape[0], 1, 1)),
            context=h_new,
        )

        embeddings, weights = embeddings.squeeze(1), weights.squeeze(1)

        embedding_out = accumulator.clone()
        embedding_out[mask] *= self.accumulator_decay
        embedding_out[mask] += embeddings

        return embedding_out, h
