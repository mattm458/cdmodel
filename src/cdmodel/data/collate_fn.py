import torch
from torch import Tensor, nn

from cdmodel.common.data import ConversationBatch


def collate_fn(batch: list[ConversationBatch]):
    conv_ids_all: list[int] = []
    features_all: list[Tensor] = []
    features_original_all: list[Tensor] = []
    features_d_all: list[Tensor] = []
    conv_lengths_all: list[Tensor] = []
    speaker_ids_all: list[Tensor] = []
    speaker_sex_all: list[Tensor] = []

    speaker_side_all: list[Tensor] = []
    speaker_exchange_all: list[Tensor] = []
    sides_exchange_all: dict[int, list[Tensor]] = {1: [], 2: []}

    embeddings_sides_all: dict[int, list[Tensor]] = {1: [], 2: []}

    segment_embeddings_all: list[Tensor] = []
    features_sides_all: dict[int, list[Tensor]] = {1: [], 2: []}
    features_d_sides_all: dict[int, list[Tensor]] = {1: [], 2: []}
    sides_lengths_all: dict[int, list[Tensor]] = {1: [], 2: []}

    text_all: list[list[str]] = []

    has_embedding_sides = False
    for b in batch:
        conv_ids_all.extend(b.conv_ids)
        features_all.append(b.features.squeeze(0))
        features_original_all.append(b.features_original.squeeze(0))
        features_d_all.append(b.features_d.squeeze(0))
        conv_lengths_all.append(b.conv_lengths.squeeze(0))
        speaker_ids_all.append(b.speaker_ids.squeeze(0))
        speaker_sex_all.append(b.speaker_sex.squeeze(0))

        speaker_side_all.append(b.speaker_side.squeeze(0))
        speaker_exchange_all.append(b.speaker_exchange.squeeze(0))
        sides_exchange_all[1].append(b.sides_exchange[1].squeeze(0))
        sides_exchange_all[2].append(b.sides_exchange[2].squeeze(0))

        if 1 in b.embeddings_sides and 2 in b.embeddings_sides:
            has_embedding_sides = True
            embeddings_sides_all[1].append(b.embeddings_sides[1])
            embeddings_sides_all[2].append(b.embeddings_sides[2])

        features_sides_all[1].append(b.features_sides[1].squeeze(0))
        features_sides_all[2].append(b.features_sides[2].squeeze(0))
        features_d_sides_all[1].append(b.features_d_sides[1].squeeze(0))
        features_d_sides_all[2].append(b.features_d_sides[2].squeeze(0))
        sides_lengths_all[1].append(b.sides_lengths[1].squeeze(0))
        sides_lengths_all[2].append(b.sides_lengths[2].squeeze(0))

        if b.segment_embeddings is not None:
            segment_embeddings_all.append(b.segment_embeddings.squeeze(0))

        text_all.extend(b.text)

    segment_embeddings: Tensor | None = None
    if len(segment_embeddings_all) > 0:
        segment_embeddings = nn.utils.rnn.pad_sequence(
            segment_embeddings_all, batch_first=True
        )

    # Pad the list of strings
    longest_text = max([len(x) for x in text_all])
    text_all = [x + ([""] * (longest_text - len(x))) for x in text_all]

    return ConversationBatch(
        conv_ids=conv_ids_all,
        features=nn.utils.rnn.pad_sequence(features_all, batch_first=True),
        features_original=nn.utils.rnn.pad_sequence(
            features_original_all, batch_first=True
        ),
        features_d=nn.utils.rnn.pad_sequence(features_d_all, batch_first=True),
        conv_lengths=torch.stack(conv_lengths_all),
        speaker_ids=nn.utils.rnn.pad_sequence(speaker_ids_all, batch_first=True),
        speaker_sex=nn.utils.rnn.pad_sequence(speaker_sex_all, batch_first=True),
        segment_embeddings=segment_embeddings,
        speaker_side=nn.utils.rnn.pad_sequence(speaker_side_all, batch_first=True),
        speaker_exchange=nn.utils.rnn.pad_sequence(
            speaker_exchange_all, batch_first=True
        ),
        sides_exchange={
            1: nn.utils.rnn.pad_sequence(sides_exchange_all[1], batch_first=True),
            2: nn.utils.rnn.pad_sequence(sides_exchange_all[2], batch_first=True),
        },
        features_sides={
            1: nn.utils.rnn.pad_sequence(features_sides_all[1], batch_first=True),
            2: nn.utils.rnn.pad_sequence(features_sides_all[2], batch_first=True),
        },
        embeddings_sides=(
            {
                1: nn.utils.rnn.pad_sequence(embeddings_sides_all[1], batch_first=True),
                2: nn.utils.rnn.pad_sequence(embeddings_sides_all[2], batch_first=True),
            }
            if has_embedding_sides
            else {}
        ),
        features_d_sides={
            1: nn.utils.rnn.pad_sequence(features_d_sides_all[1], batch_first=True),
            2: nn.utils.rnn.pad_sequence(features_d_sides_all[2], batch_first=True),
        },
        sides_lengths={
            1: torch.stack(sides_lengths_all[1]),
            2: torch.stack(sides_lengths_all[2]),
        },
        text=text_all,
    )
