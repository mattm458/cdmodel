from typing import Final, Optional

import torch
from torch import Tensor, nn

from cdmodel.model.components.attention import AdditiveAttention


class DecoderCell(nn.Module):
    def __init__(
        self,
        in_dim: int,
        att_ctx_dim: int,
        dec_ctx_dim: int,
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
            hidden_dim=in_dim, query_dim=(h_dim * num_layers) + att_ctx_dim
        )
        self.rnn = nn.GRU(
            in_dim + dec_ctx_dim, h_dim, num_layers=num_layers, batch_first=True
        )

        linear_dim = in_dim + lin_ctx_dim
        self.linear = nn.Sequential(
            *(
                ([nn.Linear(linear_dim, linear_dim), nn.ReLU()] * (lin_num_layers - 1))
                + [nn.Linear(linear_dim, len(features))]
            )
        )

    def init(self, batch_size: int, device) -> Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(
        self,
        input: Tensor,
        h: Tensor,
        att_ctx: Tensor,
        dec_ctx: Tensor,
        lin_ctx: Tensor,
        mask: Optional[Tensor] = None,
    ):
        batch_size = input.shape[0]

        q = h.permute(1, 0, 2).reshape(batch_size, -1)
        x, weights = self.attention(
            query=torch.concat([q, att_ctx.squeeze(1)], -1), keys=input, mask=mask
        )

        x, h = self.rnn(torch.concat([x, dec_ctx], dim=-1), h)
        outputs = self.linear(torch.concat([x, lin_ctx], -1))

        return outputs, h, weights
