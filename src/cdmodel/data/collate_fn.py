from collections import defaultdict
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common import ConversationData
from cdmodel.common.role_assignment import Role


def collate_fn(batches: list[ConversationData]) -> ConversationData:
    conv_id_all: Final[list[int]] = []
    segment_features_all: Final[list[Tensor]] = []
    segment_features_delta_all: Final[list[Tensor]] = []
    embeddings_all: Final[list[Tensor]] = []
    embeddings_segment_len_all: Final[list[Tensor]] = []
    num_segments_all: Final[list[int]] = []
    speaker_id_all: Final[list[list[int]]] = []
    speaker_id_idx_all: Final[list[Tensor]] = []
    speaker_role_all: Final[list[list[Role]]] = []
    speaker_role_idx_all: Final[list[Tensor]] = []
    segment_features_sides_all: Final[defaultdict[Role, list[Tensor]]] = defaultdict(
        list
    )
    segment_features_sides_len_all: Final[defaultdict[Role, list[int]]] = defaultdict(
        list
    )
    segment_features_delta_sides_all: Final[defaultdict[Role, list[Tensor]]] = (
        defaultdict(list)
    )
    role_speaker_assignment_all: Final[list[dict[Role, int]]] = []
    role_speaker_assignment_idx_all: Final[list[dict[Role, int]]] = []

    predict_next_all: Final[list[Tensor]] = []
    history_mask_a_all: Final[list[Tensor]] = []
    history_mask_b_all: Final[list[Tensor]] = []

    transcript_all: Final[list[list[str]]] = []

    # For padding embedding segments
    longest_embedding_segment: int = 0
    longest_transcript: int = 0

    for batch in batches:
        conv_id_all.extend(batch.conv_id)
        segment_features_all.append(batch.segment_features.squeeze(0))
        segment_features_delta_all.append(batch.segment_features_delta.squeeze(0))
        embeddings_all.append(batch.embeddings)
        embeddings_segment_len_all.append(batch.embeddings_segment_len)
        num_segments_all.extend(batch.num_segments)
        speaker_id_all.extend(batch.speaker_id)
        speaker_id_idx_all.append(batch.speaker_id_idx.squeeze(0))
        speaker_role_all.extend(batch.speaker_role)
        speaker_role_idx_all.append(batch.speaker_role_idx.squeeze(0))
        role_speaker_assignment_all.extend(batch.role_speaker_assignment)
        role_speaker_assignment_idx_all.extend(batch.role_speaker_assignment_idx)
        max_embeddings_len: int = int(batch.embeddings_segment_len.max().item())
        if longest_embedding_segment < max_embeddings_len:
            longest_embedding_segment = max_embeddings_len

        for role, t in batch.segment_features_sides.items():
            segment_features_sides_all[role].append(t.squeeze(0))

        for role, t_len in batch.segment_features_sides_len.items():
            segment_features_sides_len_all[role].extend(t_len)

        for role, t in batch.segment_features_delta_sides.items():
            segment_features_delta_sides_all[role].append(t.squeeze(0))

        predict_next_all.append(batch.predict_next.squeeze(0))
        history_mask_a_all.append(batch.history_mask_a.squeeze(0))
        history_mask_b_all.append(batch.history_mask_b.squeeze(0))

        transcript_all.extend(batch.transcript)
        if len(batch.transcript[0]) > longest_transcript:
            longest_transcript = len(batch.transcript[0])

    conv_id: Final[list[int]] = conv_id_all

    segment_features: Final[Tensor] = nn.utils.rnn.pad_sequence(
        segment_features_all, batch_first=True
    )

    segment_features_delta: Final[Tensor] = nn.utils.rnn.pad_sequence(
        segment_features_delta_all, batch_first=True
    )

    segment_features_sides: Final[dict[Role, Tensor]] = {}
    for role, t_list in segment_features_sides_all.items():
        segment_features_sides[role] = nn.utils.rnn.pad_sequence(
            t_list, batch_first=True
        )

    segment_features_sides_len: Final[dict[Role, list[int]]] = dict(
        segment_features_sides_len_all
    )

    segment_features_delta_sides: Final[dict[Role, Tensor]] = {}
    for role, t_list in segment_features_delta_sides_all.items():
        segment_features_delta_sides[role] = nn.utils.rnn.pad_sequence(
            t_list, batch_first=True
        )

    # Embeddings are stored in a 3-dimensional tensor with the following dimensions:
    #
    #    (conversation segments, words, embedding dimension)
    #
    # To make it possible for all turns can be encoded in parallel, all segments from
    # all conversations are concatenated along the first axis. After encoding, it is the
    # model's responsibility to break up the result back into individual conversations.
    embeddings: Final[Tensor] = torch.cat(
        [
            F.pad(x, (0, 0, 0, longest_embedding_segment - x.shape[1]))
            for x in embeddings_all
        ],
        dim=0,
    )

    # Similarly to how the embeddings are represented, the embedding segment lengths are
    # also concatenated along a single dimension. It is the model's responsibility to divide
    # sequences of lengths into individual conversations.
    embeddings_segment_len: Final[Tensor] = torch.cat(embeddings_segment_len_all, dim=0)

    num_segments: Final[list[int]] = num_segments_all

    speaker_id: Final[list[list[int]]] = speaker_id_all
    speaker_id_idx: Final[Tensor] = nn.utils.rnn.pad_sequence(
        speaker_id_idx_all, batch_first=True
    )

    speaker_role: Final[list[list[Role]]] = speaker_role_all
    speaker_role_idx: Final[Tensor] = nn.utils.rnn.pad_sequence(
        speaker_role_idx_all, batch_first=True
    )

    role_speaker_assignment = role_speaker_assignment_all
    role_speaker_assignment_idx = role_speaker_assignment_idx_all

    predict_next = nn.utils.rnn.pad_sequence(predict_next_all, batch_first=True)
    history_mask_a = nn.utils.rnn.pad_sequence(history_mask_a_all, batch_first=True)
    history_mask_b = nn.utils.rnn.pad_sequence(history_mask_b_all, batch_first=True)

    transcript: list[list[str]] = []
    for t_i in transcript_all:
        transcript.append(t_i + ["" for _ in range(longest_transcript - len(t_i))])

    return ConversationData(
        conv_id=conv_id,
        segment_features=segment_features,
        segment_features_delta=segment_features_delta,
        segment_features_sides=segment_features_sides,
        segment_features_sides_len=segment_features_sides_len,
        segment_features_delta_sides=segment_features_delta_sides,
        embeddings=embeddings,
        embeddings_segment_len=embeddings_segment_len,
        num_segments=num_segments,
        speaker_id=speaker_id,
        speaker_id_idx=speaker_id_idx,
        speaker_role=speaker_role,
        speaker_role_idx=speaker_role_idx,
        role_speaker_assignment=role_speaker_assignment,
        role_speaker_assignment_idx=role_speaker_assignment_idx,
        predict_next=predict_next,
        history_mask_a=history_mask_a,
        history_mask_b=history_mask_b,
        transcript=transcript,
    )
