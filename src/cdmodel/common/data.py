from typing import Literal, NamedTuple

from torch import Tensor


class ConversationBatch(NamedTuple):
    features: Tensor
    conv_lengths: Tensor
    speaker_ids: Tensor
    speaker_designation: Tensor
    segment_embeddings: Tensor | None
