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
        learn_rnn_initial_state: bool,
    ):
        super().__init__()

        self.flattened = False

        self.hidden_size: Final[int] = h_dim
        self.num_layers: Final[int] = num_layers
        self.skip_conn: Final[bool] = skip_conn

        self.att_ctx_dim: Final[int] = att_ctx_dim
        self.dec_ctx_dim: Final[int] = dec_ctx_dim
        self.lin_ctx_dim: Final[int] = lin_ctx_dim

        self.attention = AdditiveAttention(
            hidden_dim=in_dim, query_dim=(h_dim * num_layers) + att_ctx_dim
        )

        self.h_initial: nn.Parameter | None = (
            nn.Parameter(torch.randn(num_layers, h_dim))
            if learn_rnn_initial_state
            else None
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
        if self.h_initial is None:
            return torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
        else:
            return self.h_initial.unsqueeze(1).expand(-1, batch_size, -1)

    def forward(
        self,
        input: Tensor,
        h: Tensor,
        att_ctx: Tensor,
        dec_ctx: Tensor,
        lin_ctx: Tensor,
        mask: Optional[Tensor] = None,
        precomputed_keys: Optional[Tensor] = None,
    ):
        if not self.flattened:
            self.rnn.flatten_parameters()
            self.flattened = True

        batch_size = input.shape[0]

        q = h.permute(1, 0, 2).reshape(batch_size, -1)

        att_in = q if self.att_ctx_dim == 0 else torch.concat([q, att_ctx], -1)
        if precomputed_keys is not None:
            att_out, att_weights = self.attention(
                query=att_in,
                keys=precomputed_keys,
                mask=mask,
                precomputed_keys=True,
            )
        else:
            att_out, att_weights = self.attention(query=att_in, keys=input, mask=mask)

        dec_in = (
            att_out
            if self.dec_ctx_dim == 0
            else torch.concat([att_out, dec_ctx], dim=-1)
        )
        x, h = self.rnn(dec_in, h)

        if self.skip_conn:
            x = x + att_out

        lin_in = x if self.lin_ctx_dim == 0 else torch.concat([x, lin_ctx], -1)
        outputs = self.linear(lin_in)

        return outputs, h, att_out, att_weights
