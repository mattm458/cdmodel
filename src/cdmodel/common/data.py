from typing import NamedTuple

from torch import Tensor


class ConversationData(NamedTuple):
    conv_id: list[int]
    segment_features: Tensor
    embeddings: Tensor
    embeddings_segment_len: Tensor
    num_segments: Tensor
