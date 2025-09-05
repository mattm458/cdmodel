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
        lin_h_dim: int,
        features: list[str],
        skip_conn: bool,
    ):
        super().__init__()

        self.hidden_size: Final[int] = h_dim
        self.num_layers: Final[int] = num_layers
        self.skip_conn: Final[bool] = skip_conn

        self.attention = AdditiveAttention(
            hidden_dim=in_dim, query_dim=(h_dim * num_layers) + att_ctx_dim
        )
        self.rnn = nn.GRU(
            in_dim + dec_ctx_dim, h_dim, num_layers=num_layers, batch_first=True
        )

        linear_input_dim = in_dim + lin_ctx_dim
        if lin_num_layers < 1:
            raise ValueError("lin_num_layers must be greater than 0")
        elif lin_num_layers == 1:
            if lin_h_dim != 0:
                raise ValueError("if lin_num_layers is 1, lin_h_dim must be 0")
            self.linear = nn.Sequential(nn.Linear(linear_input_dim, len(features)))
        elif lin_num_layers == 2:
            self.linear = nn.Sequential(
                nn.Linear(linear_input_dim, lin_h_dim),
                nn.ReLU(),
                nn.Linear(lin_h_dim, len(features)),
            )
        else:
            self.linear = nn.Sequential(
                *(
                    [nn.Linear(linear_input_dim, lin_h_dim), nn.ReLU()]
                    + (
                        [nn.Linear(lin_h_dim, lin_h_dim), nn.ReLU()]
                        * (lin_num_layers - 2)
                    )
                    + [nn.Linear(lin_h_dim, len(features))]
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
        att_out, att_weights = self.attention(
            query=torch.concat([q, att_ctx], -1), keys=input, mask=mask
        )

        x, h = self.rnn(torch.concat([att_out, dec_ctx], dim=-1), h)

        if self.skip_conn:
            x = x + att_out

        outputs = self.linear(torch.concat([x, lin_ctx], -1))

        return outputs, h, att_out, att_weights
