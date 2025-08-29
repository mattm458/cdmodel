import torch
from torch import Tensor, nn

from cdmodel.common.data import ConversationBatch


def collate_fn(batch: list[ConversationBatch]):
    features_all: list[Tensor] = []
    features_d_all: list[Tensor] = []
    conv_lengths_all: list[Tensor] = []
    speaker_ids_all: list[Tensor] = []
    speaker_rank_all: list[Tensor] = []
    segment_embeddings_all: list[Tensor] = []

    for b in batch:
        features_all.append(b.features.squeeze(0))
        features_d_all.append(b.features_d.squeeze(0))
        conv_lengths_all.append(b.conv_lengths.squeeze(0))
        speaker_ids_all.append(b.speaker_ids.squeeze(0))
        speaker_rank_all.append(b.speaker_rank.squeeze(0))

        if b.segment_embeddings is not None:
            segment_embeddings_all.append(b.segment_embeddings.squeeze(0))

    segment_embeddings: Tensor | None = None
    if len(segment_embeddings_all) > 0:
        segment_embeddings = nn.utils.rnn.pad_sequence(
            segment_embeddings_all, batch_first=True
        )

    return ConversationBatch(
        features=nn.utils.rnn.pad_sequence(features_all, batch_first=True),
        features_d=nn.utils.rnn.pad_sequence(features_d_all, batch_first=True),
        conv_lengths=torch.stack(conv_lengths_all),
        speaker_ids=nn.utils.rnn.pad_sequence(speaker_ids_all, batch_first=True),
        speaker_rank=nn.utils.rnn.pad_sequence(speaker_rank_all, batch_first=True),
        segment_embeddings=segment_embeddings,
    )
