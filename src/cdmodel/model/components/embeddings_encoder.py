from typing import Final

import torch
from torch import Tensor, nn


class EmbeddingsEncoder(nn.Module):

    def __init__(
        self, emb_style: str | None, emb_proj: bool, emb_dim: int, emb_proj_dim: int
    ):
        super().__init__()

        self.enabled: Final[bool] = emb_style is not None

        self.emb_proj: nn.Module = nn.Identity()
        if self.enabled:
            if emb_dim == 0:
                raise ValueError(
                    "emb_dim must be specified when embeddings are enabled."
                )

            if emb_proj:
                self.emb_proj = nn.Sequential(
                    nn.Linear(emb_dim, emb_proj_dim), nn.Tanh()
                )
            else:
                if emb_proj_dim != 0 and emb_proj_dim != emb_dim:
                    raise ValueError(
                        "If emb_proj is False, emb_proj_dim must equal emb_dim."
                    )
        else:
            if emb_proj:
                raise ValueError("emb_proj is True, but embeddings are disabled.")

    def forward(
        self, embeddings: Tensor | None, embeddings_len: Tensor | None = None
    ) -> Tensor | None:
        if not self.enabled:
            return None

        if embeddings is None:
            return None

        return self.emb_proj(embeddings)
