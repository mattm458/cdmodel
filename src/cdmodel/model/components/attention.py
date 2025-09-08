from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int, query_dim: int):
        super().__init__()

        self.w = nn.Linear(query_dim, hidden_dim,bias=False)
        self.u = nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.v = nn.Linear(hidden_dim, 1,bias=False)

    def precompute_keys(self, keys: Tensor) -> Tensor:
        return self.u(keys)

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        mask: Optional[Tensor] = None,
        precomputed_keys: bool = False,
    ):
        if precomputed_keys:
            scores = self.v(torch.tanh(self.w(query.unsqueeze(1)) + keys))
        else:
            scores = self.v(torch.tanh(self.w(query.unsqueeze(1)) + self.u(keys)))

        if mask is None:
            weights = F.softmax(scores, dim=1)
        else:
            zero_fill_mask = ~mask.unsqueeze(-1)
            scores = scores.masked_fill(zero_fill_mask, -torch.inf)
            weights = F.softmax(scores, dim=1)
            weights = weights.masked_fill(zero_fill_mask, 0.0)

        context = weights.permute(0, 2, 1) @ keys
        return context, weights
