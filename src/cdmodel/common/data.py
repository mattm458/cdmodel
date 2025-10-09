from typing import Literal, NamedTuple

from torch import Tensor


class ConversationBatch(NamedTuple):
    conv_ids: list[int]
    conv_lengths: Tensor
    speaker_ids: Tensor
    speaker_side: Tensor
    speaker_sex: Tensor

    features: Tensor
    features_d: Tensor

    features_sides: dict[int, Tensor]
    features_d_sides: dict[int, Tensor]
    sides_lengths: dict[int, Tensor]

    segment_embeddings: Tensor | None
