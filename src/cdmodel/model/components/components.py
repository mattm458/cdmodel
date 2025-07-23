from enum import Enum
from typing import Literal, NamedTuple, Optional

import torch
from torch import Tensor, nn

from cdmodel.model.util.util import lengths_to_mask

AttentionActivation = Enum("AttentionActivation", ["softmax", "sigmoid", "tanh"])


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.ModuleList(
            [nn.GRUCell(in_dim, hidden_dim)]
            + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(dropout)

    def get_hidden(self, batch_size: int, device) -> list[Tensor]:
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    def forward(
        self,
        encoder_input: Tensor,
        hidden: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        if len(hidden) != len(self.rnn):
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x = encoder_input
        new_hidden: list[Tensor] = []

        for i, rnn in enumerate(self.rnn):
            h_out = rnn(x, hidden[i])
            x = h_out
            new_hidden.append(h_out)

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

        return x, new_hidden


class AttentionModule(nn.Module):
    pass


class AttentionScores(NamedTuple):
    a_scores: Tensor | None = None
    b_scores: Tensor | None = None
    combined_scores: Tensor | None = None


class Attention(AttentionModule):
    def __init__(
        self,
        history_in_dim: int,
        context_dim: int,
        att_dim: int,
        weighting_strategy: Literal["att"] | Literal["random"] | Literal["uniform"],
        scoring_activation: AttentionActivation = AttentionActivation.softmax,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.scoring_activation = scoring_activation
        self.weighting_strategy = weighting_strategy

        print(scoring_activation)

        # If we're using an alternative weighting strategy, we don't need weights
        if self.weighting_strategy == "att":
            print("Attention: Using normal attention scoring")
            self.history = nn.Linear(history_in_dim, att_dim, bias=False)
            self.context = nn.Linear(context_dim, att_dim, bias=False)
            self.v = nn.Linear(att_dim, 1, bias=False)
        else:
            print(f"Attention: Using {weighting_strategy} weighting strategy")

    def forward(
        self,
        history: Tensor,
        context: Tensor,
        mask: Optional[Tensor] = None,
        weight_offset: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        score: Tensor

        if self.weighting_strategy == "uniform":
            score = torch.ones(
                (history.shape[0], history.shape[1], 1),
                dtype=history.dtype,
                device=history.device,
            )
        elif self.weighting_strategy == "random":
            score = torch.rand(
                (history.shape[0], history.shape[1], 1),
                dtype=history.dtype,
                device=history.device,
            )
        elif self.weighting_strategy == "att":
            history_att: Tensor = self.history(history)
            context_att: Tensor = self.context(context).unsqueeze(1)

            score = self.v(torch.tanh(history_att + context_att))

        if mask is not None:
            score = score.masked_fill(mask, float("-inf"))

        if self.scoring_activation == AttentionActivation.softmax:
            score = torch.softmax(score, dim=1)
        elif self.scoring_activation == AttentionActivation.sigmoid:
            score = torch.sigmoid(score)
        elif self.scoring_activation == AttentionActivation.tanh:
            score = torch.tanh(score)

        if mask is not None:
            score = score.masked_fill(mask, 0.0)

        score = score.swapaxes(1, 2)

        if weight_offset is not None:
            score += weight_offset

        att_applied = torch.bmm(score, history).squeeze(1)

        if self.training:
            return att_applied, None

        return att_applied, score


class MultiheadAttention(AttentionModule):
    def __init__(self, history_dim: int, query_dim: int):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=1,
            batch_first=True,
            kdim=history_dim,
            vdim=history_dim,
        )

    def forward(
        self, history: Tensor, query: Tensor, mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor | None]:
        encoded, weights = self.att(
            query=query.unsqueeze(1),
            key=history,
            value=history,
            key_padding_mask=mask,
            need_weights=not self.training,
        )

        return encoded.nan_to_num(0).squeeze(1), (
            weights.nan_to_num(0) if weights is not None else None
        )


class DualAttention(AttentionModule):
    def __init__(
        self,
        history_in_dim: int,
        context_dim: int,
        att_dim: int,
        weighting_strategy: Literal["att"] | Literal["random"] | Literal["uniform"],
    ):
        super().__init__()

        self.our_attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
            weighting_strategy=weighting_strategy,
        )
        self.their_attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
            weighting_strategy=weighting_strategy,
        )

    def forward(
        self, history: Tensor, context: Tensor, a_mask: Tensor, b_mask: Tensor
    ) -> tuple[Tensor, AttentionScores]:
        a_att, a_scores = self.our_attention(
            history,
            context=context,
            mask=~a_mask.unsqueeze(-1),
        )

        b_att, b_scores = self.their_attention(
            history,
            context=context,
            mask=~b_mask.unsqueeze(-1),
        )

        return torch.cat([a_att, b_att], dim=-1), AttentionScores(
            a_scores=a_scores,
            b_scores=b_scores,
        )


class DualMultiheadAttention(AttentionModule):
    def __init__(self, history_dim: int, context_dim: int):
        super().__init__()

        self.our_attention = MultiheadAttention(
            history_dim=history_dim, query_dim=context_dim
        )
        self.their_attention = MultiheadAttention(
            history_dim=history_dim, query_dim=context_dim
        )

    def forward(
        self, history: Tensor, context: Tensor, a_mask: Tensor, b_mask: Tensor
    ) -> tuple[Tensor, AttentionScores]:
        a_att, a_scores = self.our_attention(
            history=history,
            query=context,
            mask=~a_mask,
        )

        b_att, b_scores = self.their_attention(
            history=history,
            query=context,
            mask=~b_mask,
        )

        return torch.cat([a_att, b_att], dim=-1), AttentionScores(
            a_scores=a_scores,
            b_scores=b_scores,
        )


class SingleAttention(AttentionModule):
    def __init__(
        self,
        history_in_dim: int,
        context_dim: int,
        att_dim: int,
        weighting_strategy: Literal["att"] | Literal["random"] | Literal["uniform"],
    ):
        super().__init__()

        self.attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
            weighting_strategy=weighting_strategy,
        )

    def forward(
        self, history: Tensor, context: Tensor, a_mask: Tensor, b_mask: Tensor
    ) -> tuple[Tensor, AttentionScores]:
        att, scores = self.attention(
            history,
            context=context,
            mask=~(a_mask + b_mask).unsqueeze(-1),
        )

        return att, AttentionScores(combined_scores=scores)


class SingleMultiheadAttention(AttentionModule):
    def __init__(self, history_dim: int, context_dim: int):
        super().__init__()

        self.attention = MultiheadAttention(
            history_dim=history_dim, query_dim=context_dim
        )

    def forward(
        self, history: Tensor, context: Tensor, a_mask: Tensor, b_mask: Tensor
    ) -> tuple[Tensor, AttentionScores]:
        att, scores = self.attention(
            history=history, query=context, mask=~(a_mask + b_mask)
        )

        return att, AttentionScores(combined_scores=scores)


class SinglePartnerAttention(AttentionModule):
    def __init__(
        self,
        history_in_dim: int,
        context_dim: int,
        att_dim: int,
        weighting_strategy: Literal["att"] | Literal["random"] | Literal["uniform"],
    ):
        super().__init__()

        self.attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
            weighting_strategy=weighting_strategy,
        )

    def forward(
        self, history: Tensor, context: Tensor, a_mask: Tensor, b_mask: Tensor
    ) -> tuple[Tensor, AttentionScores]:
        att, scores = self.attention(
            history,
            context=context,
            mask=~(b_mask).unsqueeze(-1),
        )

        return att, AttentionScores(b_scores=scores)


class SinglePartnerMultiheadAttention(AttentionModule):
    def __init__(
        self,
        history_in_dim: int,
        context_dim: int,
    ):
        super().__init__()

        self.attention = MultiheadAttention(
            history_dim=history_in_dim,
            query_dim=context_dim,
        )

    def forward(
        self, history: Tensor, context: Tensor, a_mask: Tensor, b_mask: Tensor
    ) -> tuple[Tensor, AttentionScores]:
        att, scores = self.attention(history=history, query=context, mask=~b_mask)

        return att, AttentionScores(b_scores=scores)


class NoopAttention(AttentionModule):
    def forward(
        self, history: Tensor, context: Tensor, a_mask: Tensor, b_mask: Tensor
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        return history[:, -1], (None, None)


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        encoder_out_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        attention_dim: int,
        pack_sequence=True,
    ):
        super().__init__()

        lstm_out_dim = encoder_out_dim // 2

        self.encoder_out_dim = encoder_out_dim
        self.encoder_num_layers = encoder_num_layers
        self.pack_sequence = pack_sequence

        self.encoder = nn.GRU(
            embedding_dim,
            lstm_out_dim,
            bidirectional=True,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            batch_first=True,
        )
        self.encoder.flatten_parameters()

        self.attention = Attention(
            history_in_dim=encoder_out_dim,
            context_dim=encoder_out_dim * 2,
            att_dim=attention_dim,
            weighting_strategy="att",
        )

    def forward(self, encoder_in: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = encoder_in.shape[0]

        if self.pack_sequence:
            encoder_in_packed = nn.utils.rnn.pack_padded_sequence(
                encoder_in,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            encoder_out_tmp, h = self.encoder(encoder_in_packed)

            encoder_out, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_out_tmp, batch_first=True
            )
        else:
            encoder_out, h = self.encoder(encoder_in)

        h = h.swapaxes(0, 1).reshape(batch_size, -1)

        return self.attention(
            history=encoder_out,
            context=h,
            mask=lengths_to_mask(lengths, encoder_out.shape[1]),
        )


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_in_dim: int,
        decoder_dropout: float,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: Literal["tanh", None],
        output_layers: int,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.rnn = nn.ModuleList(
            [nn.GRUCell(decoder_in_dim, hidden_dim)]
            + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(decoder_dropout)

        linear_arr: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim if output_layers > 1 else output_dim)
        ]
        for i in range(output_layers - 1):
            if i == output_layers - 2:
                linear_arr.extend([nn.ELU(), nn.Linear(hidden_dim, output_dim)])
            else:
                linear_arr.extend([nn.ELU(), nn.Linear(hidden_dim, hidden_dim)])

        if activation == "tanh":
            print("Decoder: Tanh activation")
            linear_arr.append(nn.Tanh())

        self.linear = nn.Sequential(*linear_arr)

    def get_hidden(self, batch_size: int, device) -> list[Tensor]:
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    def forward(
        self,
        encoded: Tensor,
        hidden: list[Tensor],
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        new_hidden: list[Tensor] = []

        x = encoded

        for i, rnn in enumerate(self.rnn):
            h_out = rnn(x, hidden[i])
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            new_hidden.append(h_out)

        out = self.linear(x)

        return out, new_hidden, x
