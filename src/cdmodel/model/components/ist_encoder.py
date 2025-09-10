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
        h_dim: int,
        layers: int,
        learn_rnn_initial_state: bool,
    ):
        super().__init__()

        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        print("tokens", self.tokens.shape)

        self.h_initial: nn.Parameter | None = (
            nn.Parameter(torch.randn(layers * 2, 1, h_dim // (layers * 2)))
            if learn_rnn_initial_state
            else None
        )

        self.rnn = nn.GRU(
            in_dim,
            h_dim // (layers * 2),
            bidirectional=True,
            batch_first=True,
            num_layers=layers,
        )

        self.attention = AdditiveAttention(hidden_dim=token_dim, query_dim=h_dim)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, rnn_h = self.rnn(
            x_packed,
            (
                self.h_initial.expand(-1, batch_size, -1)
                if self.h_initial is not None
                else None
            ),
        )

        rnn_h = rnn_h.swapaxes(0, 1).reshape(batch_size, -1)
        ist, w = self.attention(
            query=rnn_h, keys=F.tanh(self.tokens)[None, :, :].expand(batch_size, -1, -1)
        )
        print(w.shape, w.min(1)[0], w.max(1)[0])
        return ist.squeeze(1), w
