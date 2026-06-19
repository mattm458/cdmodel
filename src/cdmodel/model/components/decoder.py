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

        self.hidden_size: Final[int] = h_dim
        self.num_layers: Final[int] = num_layers
        self.skip_conn: Final[bool] = skip_conn

        self.att_ctx_dim: Final[int] = att_ctx_dim
        self.dec_ctx_dim: Final[int] = dec_ctx_dim
        self.lin_ctx_dim: Final[int] = lin_ctx_dim

        self.query_proj = nn.Sequential(
            nn.Linear(h_dim + att_ctx_dim, h_dim), nn.Tanh()
        )
        self.attention = AdditiveAttention(hidden_dim=h_dim, query_dim=h_dim)

        if learn_rnn_initial_state:
            print("Decoder: Learning RNN initial state")
        self.h_initial = nn.Parameter(
            (
                torch.randn(num_layers, 1, in_dim)
                if learn_rnn_initial_state
                else torch.zeros(num_layers, 1, in_dim)
            ),
            requires_grad=learn_rnn_initial_state,
        )

        self.rnn = nn.GRU(
            in_dim + dec_ctx_dim, in_dim, num_layers=num_layers, batch_first=True
        )

        linear_input_dim = in_dim + lin_ctx_dim
        if lin_num_layers < 1:
            raise ValueError("lin_num_layers must be greater than 0")
        elif lin_num_layers == 1:
            if lin_h_dim != 0:
                raise ValueError("if lin_num_layers is 1, lin_h_dim must be 0")
            self.out = nn.Sequential(nn.Linear(linear_input_dim, len(features)))
        else:
            linear_layers = [nn.Linear(linear_input_dim, lin_h_dim), nn.ELU()]
            for _ in range(lin_num_layers - 2):
                linear_layers.extend([nn.Linear(lin_h_dim, lin_h_dim), nn.ELU()])
            linear_layers.extend([nn.Linear(lin_h_dim, len(features))])
            self.out = nn.Sequential(*linear_layers)

    def init(self, batch_size: int, device) -> Tensor:
        return self.h_initial.expand(-1, batch_size, -1).contiguous()

    def forward(
        self,
        input: Tensor,
        h: Tensor,
        att_ctx: Tensor,
        dec_ctx: Tensor,
        lin_ctx: Tensor,
        encoder_skip: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        precomputed_keys: Optional[Tensor] = None,
    ):
        q = h[-1]
        att_in = self.query_proj(
            q if self.att_ctx_dim == 0 else torch.concat([q, att_ctx], -1)
        )

        if precomputed_keys is not None:
            att_out, att_weights = self.attention(
                query=att_in,
                keys=precomputed_keys,
                mask=mask,
                precomputed_keys=True,
            )
        else:
            att_out, att_weights = self.attention(query=att_in, keys=input, mask=mask)

        if encoder_skip is not None:
            att_out = att_out + encoder_skip

        dec_in = (
            att_out
            if self.dec_ctx_dim == 0
            else torch.concat([att_out, dec_ctx], dim=-1)
        )

        x, h = self.rnn(dec_in, h)

        if self.skip_conn:
            x = x + att_out

        lin_in = x if self.lin_ctx_dim == 0 else torch.concat([x, lin_ctx], -1)
        outputs = self.out(lin_in)

        return outputs, h, (att_weights,)


class NoneAttentionDecoderCell(nn.Module):
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

        self.hidden_size: Final[int] = h_dim
        self.num_layers: Final[int] = num_layers
        self.skip_conn: Final[bool] = skip_conn

        self.att_ctx_dim: Final[int] = att_ctx_dim
        self.dec_ctx_dim: Final[int] = dec_ctx_dim
        self.lin_ctx_dim: Final[int] = lin_ctx_dim

        if learn_rnn_initial_state:
            print("Decoder: Learning RNN initial state")
        self.h_initial = nn.Parameter(
            (
                torch.randn(num_layers, 1, in_dim)
                if learn_rnn_initial_state
                else torch.zeros(num_layers, 1, in_dim)
            ),
            requires_grad=learn_rnn_initial_state,
        )

        self.query_proj = nn.Sequential(
            nn.Linear(h_dim + att_ctx_dim, h_dim), nn.Tanh()
        )
        self.rnn = nn.GRU(
            in_dim + dec_ctx_dim, in_dim, num_layers=num_layers, batch_first=True
        )

        linear_input_dim = in_dim + lin_ctx_dim
        if lin_num_layers < 1:
            raise ValueError("lin_num_layers must be greater than 0")
        elif lin_num_layers == 1:
            if lin_h_dim != 0:
                raise ValueError("if lin_num_layers is 1, lin_h_dim must be 0")
            self.out = nn.Sequential(nn.Linear(linear_input_dim, len(features)))
        else:
            linear_layers = [nn.Linear(linear_input_dim, lin_h_dim), nn.ELU()]
            for _ in range(lin_num_layers - 2):
                linear_layers.extend([nn.Linear(lin_h_dim, lin_h_dim), nn.ELU()])
            linear_layers.extend([nn.Linear(lin_h_dim, len(features))])
            self.out = nn.Sequential(*linear_layers)

    def init(self, batch_size: int, device) -> Tensor:
        return self.h_initial.expand(-1, batch_size, -1).contiguous()

    def forward(
        self,
        input: Tensor,
        h: Tensor,
        att_ctx: Tensor,
        dec_ctx: Tensor,
        lin_ctx: Tensor,
        encoder_skip: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        precomputed_keys: Optional[Tensor] = None,
    ):
        att_out = self.query_proj(input[mask][:, None, :])

        if encoder_skip is not None:
            att_out = att_out + encoder_skip

        dec_in = (
            att_out
            if self.dec_ctx_dim == 0
            else torch.concat([att_out, dec_ctx], dim=-1)
        )

        x, h = self.rnn(dec_in, h)

        if self.skip_conn:
            x = x + att_out

        lin_in = x if self.lin_ctx_dim == 0 else torch.concat([x, lin_ctx], -1)
        outputs = self.out(lin_in)

        return outputs, h, (None,)


class DualAttentionDecoderCell(nn.Module):
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

        self.hidden_size: Final[int] = h_dim
        self.num_layers: Final[int] = num_layers
        self.skip_conn: Final[bool] = skip_conn

        self.att_ctx_dim: Final[int] = att_ctx_dim
        self.dec_ctx_dim: Final[int] = dec_ctx_dim
        self.lin_ctx_dim: Final[int] = lin_ctx_dim

        self.query_self_proj = nn.Sequential(
            nn.Linear(h_dim + att_ctx_dim, h_dim), nn.Tanh()
        )
        self.attention_self = AdditiveAttention(hidden_dim=in_dim, query_dim=h_dim)

        self.query_partner_proj = nn.Sequential(
            nn.Linear(h_dim + att_ctx_dim, h_dim), nn.Tanh()
        )
        self.attention_partner = AdditiveAttention(hidden_dim=in_dim, query_dim=h_dim)

        if learn_rnn_initial_state:
            print("Decoder: Learning RNN initial state")
        self.h_initial = nn.Parameter(
            (
                torch.randn(num_layers, 1, in_dim)
                if learn_rnn_initial_state
                else torch.zeros(num_layers, 1, in_dim)
            ),
            requires_grad=learn_rnn_initial_state,
        )

        self.rnn = nn.GRU(
            (in_dim * 2) + dec_ctx_dim, in_dim, num_layers=num_layers, batch_first=True
        )

        linear_input_dim = in_dim + lin_ctx_dim
        if lin_num_layers < 1:
            raise ValueError("lin_num_layers must be greater than 0")
        elif lin_num_layers == 1:
            if lin_h_dim != 0:
                raise ValueError("if lin_num_layers is 1, lin_h_dim must be 0")
            self.out = nn.Sequential(nn.Linear(linear_input_dim, len(features)))
        elif lin_num_layers == 2:
            self.out = nn.Sequential(
                nn.Linear(linear_input_dim, lin_h_dim),
                nn.ReLU(),
                nn.Linear(lin_h_dim, len(features)),
            )
        else:
            linear_layers = [nn.Linear(linear_input_dim, lin_h_dim), nn.ReLU()]
            for _ in range(lin_num_layers - 2):
                linear_layers += [nn.Linear(lin_h_dim, lin_h_dim), nn.ReLU()]
            linear_layers += [nn.Linear(lin_h_dim, len(features))]
            self.out = nn.Sequential(*linear_layers)

    def init(self, batch_size: int, device) -> Tensor:
        return self.h_initial.expand(-1, batch_size, -1).contiguous()

    def forward(
        self,
        input: Tensor,
        h: Tensor,
        att_ctx: Tensor,
        dec_ctx: Tensor,
        lin_ctx: Tensor,
        mask: tuple[Tensor, Tensor],
        encoder_skip: Optional[Tensor] = None,
        precomputed_keys: Optional[Tensor] = None,
    ):
        if precomputed_keys is not None:
            raise Exception(
                "DualAttentionDecoderCell is incompatible with precomputed keys"
            )

        partner_mask, self_mask = mask

        q = h[-1]
        att_in = q if self.att_ctx_dim == 0 else torch.concat([q, att_ctx], -1)

        partner_att_out, partner_att_weights = self.attention_partner(
            query=self.query_partner_proj(att_in), keys=input, mask=partner_mask
        )
        self_att_out, self_att_weights = self.attention_self(
            query=self.query_self_proj(att_in), keys=input, mask=self_mask
        )

        if encoder_skip is not None:
            self_att_out = self_att_out + encoder_skip
            partner_att_out = partner_att_out + encoder_skip

        dec_in = (
            torch.concat([self_att_out, partner_att_out], dim=-1)
            if self.dec_ctx_dim == 0
            else torch.concat([self_att_out, partner_att_out, dec_ctx], dim=-1)
        )
        x, h = self.rnn(dec_in, h)

        if self.skip_conn:
            x = x + self_att_out + partner_att_out

        lin_in = x if self.lin_ctx_dim == 0 else torch.concat([x, lin_ctx], -1)
        outputs = self.out(lin_in)

        return outputs, h, (partner_att_weights, self_att_weights)


class FusedDualAttentionDecoderCell(nn.Module):
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

        self.hidden_size: Final[int] = h_dim
        self.num_layers: Final[int] = num_layers
        self.skip_conn: Final[bool] = skip_conn

        self.att_ctx_dim: Final[int] = att_ctx_dim
        self.dec_ctx_dim: Final[int] = dec_ctx_dim
        self.lin_ctx_dim: Final[int] = lin_ctx_dim

        self.query_self_proj = nn.Sequential(
            nn.Linear(h_dim + att_ctx_dim, h_dim), nn.Tanh()
        )
        self.attention_self = AdditiveAttention(hidden_dim=in_dim, query_dim=h_dim)

        self.query_partner_proj = nn.Sequential(
            nn.Linear(h_dim + att_ctx_dim, h_dim), nn.Tanh()
        )
        self.attention_partner = AdditiveAttention(hidden_dim=in_dim, query_dim=h_dim)

        self.query_fusion_proj = nn.Sequential(
            nn.Linear(h_dim + att_ctx_dim, h_dim), nn.Tanh()
        )
        self.attention_fusion = AdditiveAttention(hidden_dim=in_dim, query_dim=h_dim)

        if learn_rnn_initial_state:
            print("Decoder: Learning RNN initial state")
        self.h_initial = nn.Parameter(
            (
                torch.randn(num_layers, 1, in_dim)
                if learn_rnn_initial_state
                else torch.zeros(num_layers, 1, in_dim)
            ),
            requires_grad=learn_rnn_initial_state,
        )

        self.rnn = nn.GRU(
            in_dim + dec_ctx_dim, in_dim, num_layers=num_layers, batch_first=True
        )

        linear_input_dim = in_dim + lin_ctx_dim
        if lin_num_layers < 1:
            raise ValueError("lin_num_layers must be greater than 0")
        elif lin_num_layers == 1:
            if lin_h_dim != 0:
                raise ValueError("if lin_num_layers is 1, lin_h_dim must be 0")
            self.out = nn.Sequential(nn.Linear(linear_input_dim, len(features)))
        elif lin_num_layers == 2:
            self.out = nn.Sequential(
                nn.Linear(linear_input_dim, lin_h_dim),
                nn.ReLU(),
                nn.Linear(lin_h_dim, len(features)),
            )
        else:
            linear_layers = [nn.Linear(linear_input_dim, lin_h_dim), nn.ReLU()]
            for _ in range(lin_num_layers - 2):
                linear_layers += [nn.Linear(lin_h_dim, lin_h_dim), nn.ReLU()]
            linear_layers += [nn.Linear(lin_h_dim, len(features))]
            self.out = nn.Sequential(*linear_layers)

    def init(self, batch_size: int, device) -> Tensor:
        return self.h_initial.expand(-1, batch_size, -1).contiguous()

    def forward(
        self,
        input: Tensor,
        h: Tensor,
        att_ctx: Tensor,
        dec_ctx: Tensor,
        lin_ctx: Tensor,
        mask: tuple[Tensor, Tensor],
        encoder_skip: Optional[Tensor] = None,
        precomputed_keys: Optional[Tensor] = None,
    ):
        if precomputed_keys is not None:
            raise Exception(
                "DualAttentionDecoderCell is incompatible with precomputed keys"
            )

        partner_mask, self_mask = mask

        q = h[-1]
        q = q if self.att_ctx_dim == 0 else torch.concat([q, att_ctx], -1)

        partner_att_out, partner_att_weights = self.attention_partner(
            query=self.query_partner_proj(q), keys=input, mask=partner_mask
        )
        self_att_out, self_att_weights = self.attention_self(
            query=self.query_self_proj(q), keys=input, mask=self_mask
        )

        att_fusion_in = torch.concat([self_att_out, partner_att_out], dim=1)
        fused_att_out, fused_att_weights = self.attention_fusion(
            query=self.query_fusion_proj(q), keys=att_fusion_in
        )

        if encoder_skip is not None:
            fused_att_out = fused_att_out + encoder_skip

        dec_in = (
            fused_att_out
            if self.dec_ctx_dim == 0
            else torch.concat([fused_att_out, dec_ctx], dim=-1)
        )
        x, h = self.rnn(dec_in, h)

        if self.skip_conn:
            x = x + self_att_out + partner_att_out

        lin_in = x if self.lin_ctx_dim == 0 else torch.concat([x, lin_ctx], -1)
        outputs = self.out(lin_in)

        return outputs, h, (partner_att_weights, self_att_weights)
