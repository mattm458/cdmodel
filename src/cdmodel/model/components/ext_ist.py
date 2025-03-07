from typing import Final, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.model.components.components import Attention, AttentionActivation


class ISTOneShotEncoder(nn.Module):
    def __init__(
        self,
        token_count: int,
        token_dim: int,
        num_features: int,
        att_dim: int,
        att_activation: AttentionActivation,
        ext_ist_use_feature_deltas: bool,
        ext_ist_use_feature_values: bool,
        ext_ist_tanh_pre: bool,
        ext_ist_tanh_post: bool,
    ):
        super().__init__()

        self.ext_ist_use_feature_deltas = ext_ist_use_feature_deltas
        self.ext_ist_use_feature_values = ext_ist_use_feature_values

        if ext_ist_tanh_pre:
            print("ISTOneShotEncoder: Applying tanh to tokens")
        else:
            print("ISTOneShotEncoder: Not applying tanh to tokens")

        if ext_ist_tanh_post:
            print("ISTOneShotEncoder: Applying tanh to IST attention output")
        else:
            print("ISTOneShotEncoder: Not applying tanh to IST attention output")

        self.ext_ist_tanh_pre = ext_ist_tanh_pre
        self.ext_ist_tanh_post = ext_ist_tanh_post

        num_features = num_features * (
            ext_ist_use_feature_deltas + ext_ist_use_feature_values
        )

        self.tokens = nn.Parameter(
            torch.randn(token_count, token_dim), requires_grad=True
        )
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
        self,
        feature_deltas: Tensor,
        feature_values: Tensor,
        features_len: Tensor,
        offset: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size: Final[int] = feature_deltas.shape[0]

        features_arr: list[Tensor] = []
        if self.ext_ist_use_feature_deltas:
            features_arr.append(feature_deltas)
        if self.ext_ist_use_feature_values:
            features_arr.append(feature_values)

        features: Tensor = torch.cat(features_arr, dim=-1)

        _, h = self.encoder(
            nn.utils.rnn.pack_padded_sequence(
                features, features_len, batch_first=True, enforce_sorted=False
            )
        )

        tokens = self.tokens
        if self.ext_ist_tanh_pre:
            tokens = F.tanh(tokens)

        embeddings, weights = self.attention(
            history=tokens.repeat((batch_size, 1, 1)),
            context=torch.cat([h[0], h[1]], dim=-1),
            weight_offset=offset,
        )

        if self.ext_ist_tanh_post:
            embeddings = F.tanh(embeddings)

        return embeddings.squeeze(1), weights.squeeze(1)


class ISTIncrementalEncoder(nn.Module):
    def __init__(
        self,
        token_count: int,
        token_dim: int,
        num_features: int,
        att_dim: int,
        att_activation: AttentionActivation,
        ext_ist_tanh_pre: bool,
        ext_ist_tanh_post: bool,
        accumulator_decay: float = 1.0,
    ):
        super().__init__()

        self.accumulator_decay: Final[float] = 1.0 - accumulator_decay
        self.ext_ist_tanh_pre = ext_ist_tanh_pre
        self.ext_ist_tanh_post = ext_ist_tanh_post

        self.token_dim: Final[int] = token_dim

        self.tokens = nn.Parameter(
            torch.randn(token_count, token_dim), requires_grad=True
        )
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
    ) -> tuple[Tensor, Tensor]:
        h = h.clone()

        h_new = self.encoder(features[mask], h[mask])
        h[mask] = h_new

        tokens = self.tokens
        if self.ext_ist_tanh_pre:
            tokens = F.tanh(tokens)

        embeddings, weights = self.attention(
            history=tokens.repeat((h_new.shape[0], 1, 1)),
            context=h_new,
        )

        embeddings, weights = embeddings.squeeze(1), weights.squeeze(1)

        embedding_out = accumulator.clone()
        if self.ext_ist_tanh_post:
            embedding_out = F.tanh(embedding_out)

        embedding_out[mask] *= self.accumulator_decay
        embedding_out[mask] += embeddings

        return embedding_out, h
