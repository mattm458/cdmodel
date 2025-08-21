from dataclasses import dataclass
from typing import Final, Optional

import torch
from torch import Tensor, nn

from cdmodel.model.components.attention import AdditiveAttention


@dataclass
class DecoderState:
    h: Tensor


class DecoderCell(nn.Module):
    def __init__(
        self,
        in_dim: int,
        att_ctx_dim: int,
        ctx_dim: int,
        lin_ctx_dim: int,
        h_dim: int,
        num_layers: int,
        lin_num_layers: int,
        features: list[str],
    ):
        super().__init__()

        self.hidden_size: Final[int] = h_dim
        self.num_layers: Final[int] = num_layers

        self.attention = AdditiveAttention(
            hidden_dim=in_dim,
            query_dim=(h_dim * num_layers) + att_ctx_dim,
        )
        self.rnn = nn.GRU(
            in_dim + ctx_dim,
            h_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        linear_dim = in_dim + lin_ctx_dim
        self.linear = nn.Sequential(
            *(
                ([nn.Linear(linear_dim, linear_dim), nn.ReLU()] * (lin_num_layers - 1))
                + [nn.Linear(linear_dim, len(features))]
            )
        )

    def init(self, batch_size: int, device) -> DecoderState:
        return DecoderState(
            h=torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )

    def forward(
        self,
        state: DecoderState,
        input: Tensor,
        att_ctx: Optional[Tensor] = None,
        dec_ctx: Optional[Tensor] = None,
        lin_ctx: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        batch_size = input.shape[0]

        query = state.h.permute(1, 0, 2).reshape(batch_size, -1)

        if att_ctx is not None:
            query = torch.cat([query, att_ctx.squeeze(1)], -1)

        x, weights = self.attention(query=query, keys=input, mask=mask)

        if dec_ctx is not None:
            x = torch.concat([x, dec_ctx], dim=-1)

        x, state.h = self.rnn(x, state.h)

        if lin_ctx is not None:
            x = torch.cat([x, lin_ctx], -1)

        outputs = self.linear(x)
        return outputs, weights
