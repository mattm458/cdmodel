import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Final
from cdmodel.model.components.attention import AdditiveAttention


class ISTEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_tokens: int,
        token_dim: int,
        num_layers: int,
        learn_rnn_initial_state: bool,
        heads: int,
        bidirectional: bool,
    ):
        super().__init__()

        if num_tokens <= 0:
            raise Exception("Number of tokens must be greater than 0")
        elif num_tokens == 1:
            pass
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=heads,
                batch_first=True,
            )

            self.tokens = nn.Parameter(torch.zeros(num_tokens, token_dim))
            nn.init.normal_(self.tokens, mean=0, std=0.5)
            print(f"ISTEncoder: Using {self.tokens.shape} tokens")

        self.num_tokens: Final[int] = num_tokens
        self.bidirectional: Final[bool] = bidirectional
        self.num_layers: Final[int] = num_layers

        if learn_rnn_initial_state:
            print("ISTEncoder: Learning RNN initial state")
        self.h_initial = nn.Parameter(
            (
                (
                    torch.randn(num_layers * 2, 1, token_dim // 2)
                    if bidirectional
                    else torch.randn(num_layers, 1, token_dim)
                )
                if learn_rnn_initial_state
                else (
                    torch.zeros(num_layers * 2, 1, token_dim // 2)
                    if bidirectional
                    else torch.zeros(num_layers, 1, token_dim)
                )
            ),
            requires_grad=learn_rnn_initial_state,
        )
        self.rnn = nn.GRU(
            in_dim,
            token_dim // 2 if bidirectional else token_dim,
            bidirectional=bidirectional,
            batch_first=True,
            num_layers=num_layers,
        )

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor | None]:
        batch_size = x.shape[0]

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, rnn_h = self.rnn(x_packed, self.h_initial.expand(-1, batch_size, -1))

        rnn_h = (
            rnn_h.view(self.num_layers, 2 if self.bidirectional else 1, batch_size, -1)[
                -1
            ]
            .swapaxes(0, 1)
            .reshape(batch_size, -1)
        )

        if self.num_tokens > 1:
            query = rnn_h.unsqueeze(1)
            tokens = F.tanh(self.tokens)[None, :, :].expand(batch_size, -1, -1)
            ist, w = self.attention(query=query, key=tokens, value=tokens)
            ist = ist.squeeze(1)

        else:
            ist = rnn_h
            w = None

        return ist, w
