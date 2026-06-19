from typing import NamedTuple

from torch import Tensor


class ConversationBatch(NamedTuple):
    conv_ids: list[int]
    conv_lengths: Tensor
    speaker_ids: Tensor
    speaker_sex: Tensor

    speaker_side: Tensor
    speaker_exchange: Tensor
    sides_exchange: dict[int, Tensor]

    features: Tensor
    features_original: Tensor
    features_d: Tensor

    features_sides: dict[int, Tensor]
    embeddings_sides: dict[int, Tensor]
    features_d_sides: dict[int, Tensor]
    sides_lengths: dict[int, Tensor]

    segment_embeddings: Tensor | None
    text: list[list[str]]
