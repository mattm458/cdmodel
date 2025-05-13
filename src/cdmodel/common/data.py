from enum import Enum
from typing import NamedTuple

from torch import Tensor

from cdmodel.common.role_assignment import Role


class ConversationData(NamedTuple):
    # Metadata
    conv_id: list[int]
    num_segments: list[int]

    # Turn-level features
    segment_features: Tensor
    segment_features_delta: Tensor
    segment_features_sides: dict[Role, Tensor]
    segment_features_sides_len: dict[Role, list[int]]
    segment_features_delta_sides: dict[Role, Tensor]
    transcript: list[list[str]]

    # Speaker metadata
    predict_next: Tensor
    history_mask_a: Tensor
    history_mask_b: Tensor

    # Turn-level word embeddings
    embeddings: Tensor
    embeddings_segment_len: Tensor

    # Speaker data
    speaker_id: list[list[int]]
    speaker_id_idx: Tensor
    speaker_role: list[list[Role | None]]
    speaker_role_idx: Tensor
    role_speaker_assignment: list[dict[Role, int]]
    role_speaker_assignment_idx: list[dict[Role, int]]
