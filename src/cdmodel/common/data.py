from typing import Literal, NamedTuple

from torch import Tensor


class ConversationBatch(NamedTuple):
    conv_ids: list[int]
    features: Tensor
    features_sides: dict[int, Tensor]
    features_d: Tensor
    features_d_sides: dict[int, Tensor]
    sides_lengths: dict[int, Tensor]
    conv_lengths: Tensor
    speaker_ids: Tensor
    speaker_side: Tensor
    segment_embeddings: Tensor | None
