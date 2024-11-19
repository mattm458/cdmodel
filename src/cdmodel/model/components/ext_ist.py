from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.model.components.components import Attention


class ISTOneShotEncoder(nn.Module):
    def __init__(
        self, token_count: int, token_dim: int, num_features: int, att_dim: int
    ):
        super().__init__()

        self.tokens = nn.Parameter(
            torch.randn(token_count, token_dim), requires_grad=True
        )

        self.encoder = nn.GRU(
            num_features, token_dim // 2, bidirectional=True, batch_first=True
        )

        self.attention = Attention(
            history_in_dim=token_dim, context_dim=att_dim, att_dim=att_dim
        )

    def forward(self, features: Tensor, features_len: Tensor):
        batch_size: Final[int] = features.shape[0]

        _, h = self.encoder(
            nn.utils.rnn.pack_padded_sequence(
                features, features_len, batch_first=True, enforce_sorted=False
            )
        )

        h_arr: Final[list[Tensor]] = [h[0], h[1]]

        embeddings, weights = self.attention(
            history=F.tanh(self.tokens).repeat((batch_size, 1, 1)),
            context=torch.cat(h_arr, dim=-1),
        )

        return embeddings.squeeze(), weights.squeeze()


class ISTIncrementalEncoder(nn.Module):
    pass
