from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.common import ConversationData


def collate_fn(batches: list[ConversationData]) -> ConversationData:
    conv_id_all: Final[list[int]] = []
    segment_features_all: Final[list[Tensor]] = []
    embeddings_all: Final[list[Tensor]] = []
    embeddings_segment_len_all: Final[list[Tensor]] = []
    num_segments_all: Final[list[int]] = []
    speaker_id_all: Final[list[list[int]]] = []
    speaker_id_idx_all: Final[list[Tensor]] = []

    # For padding embedding segments
    longest_embedding_segment: int = 0

    for batch in batches:
        conv_id_all.extend(batch.conv_id)
        segment_features_all.append(batch.segment_features.squeeze(0))
        embeddings_all.append(batch.embeddings)
        embeddings_segment_len_all.append(batch.embeddings_segment_len)
        num_segments_all.extend(batch.num_segments)
        speaker_id_all.extend(batch.speaker_id)
        speaker_id_idx_all.append(batch.speaker_id_idx.squeeze(0))

        max_embeddings_len: int = int(batch.embeddings_segment_len.max().item())
        if longest_embedding_segment < max_embeddings_len:
            longest_embedding_segment = max_embeddings_len

    conv_id: Final[list[int]] = conv_id_all

    segment_features: Final[Tensor] = nn.utils.rnn.pad_sequence(
        segment_features_all, batch_first=True
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

    return ConversationData(
        conv_id=conv_id,
        segment_features=segment_features,
        embeddings=embeddings,
        embeddings_segment_len=embeddings_segment_len,
        num_segments=num_segments,
        speaker_id=speaker_id,
        speaker_id_idx=speaker_id_idx,
    )
