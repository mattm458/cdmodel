import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.model.components.attention import AdditiveAttention


class ISTEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int,
        num_layers: int,
        learn_rnn_initial_state: bool,
    ):
        super().__init__()

        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        print(f"ISTEncoder: Using {self.tokens.shape} tokens")

        if learn_rnn_initial_state:
            print("ISTEncoder: Learning RNN initial state")
        self.h_initial = nn.Parameter(
            (
                torch.randn(num_layers * 2, 1, hidden_dim)
                if learn_rnn_initial_state
                else torch.zeros(num_layers, 1, hidden_dim)
            ),
            requires_grad=learn_rnn_initial_state,
        )
        self.rnn = nn.GRU(
            in_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
        )

        self.attention = AdditiveAttention(
            hidden_dim=token_dim,
            query_dim=(hidden_dim * 2 * num_layers),
            activation="sigmoid",
        )

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, rnn_h = self.rnn(x_packed, self.h_initial.expand(-1, batch_size, -1))

        rnn_h = rnn_h.swapaxes(0, 1).reshape(batch_size, -1)
        ist, w = self.attention(
            query=rnn_h,
            keys=F.tanh(self.tokens)[None, :, :].expand(batch_size, -1, -1),
        )

        return ist.squeeze(1), w
