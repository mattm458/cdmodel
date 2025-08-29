from typing import Literal, NamedTuple

from torch import Tensor


class ConversationBatch(NamedTuple):
    features: Tensor
    features_d: Tensor
    conv_lengths: Tensor
    speaker_ids: Tensor
    speaker_rank: Tensor
    segment_embeddings: Tensor | None
