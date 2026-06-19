from os import path
from typing import Final, Literal, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from cdmodel.common.data import ConversationBatch

PrimarySpeakerSelectionStrategy = Literal["first"] | Literal["second"] | Literal["both"]
NormalizationStrategy = (
    Literal["zscore"]
    | Literal["zscore_conv"]
    | Literal["zscore_conv_speaker"]
)


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        features: list[str],
        conv_ids: list[int],
        zero_pad: bool,
        embeddings: str | None,
        normalization: NormalizationStrategy,
        primary_speaker_selection: PrimarySpeakerSelectionStrategy,
        norm_z_mean: dict[str, Tensor],
        norm_z_std: dict[str, Tensor],
    ):
        self.dataset_dir: Final[str] = dataset_dir
        self.features: list[str] = features
        self.conv_ids: Final[list[int]] = conv_ids
        self.zero_pad: Final[bool] = zero_pad
        self.embeddings: Final[str | None] = embeddings
        self.normalization: Final[str] = normalization
        self.primary_speaker_selection: Final[PrimarySpeakerSelectionStrategy] = (
            primary_speaker_selection
        )

        self.norm_z_mean: Final[dict[str, Tensor]] = norm_z_mean
        self.norm_z_std: Final[dict[str, Tensor]] = norm_z_std

    def __len__(self) -> int:
        if self.primary_speaker_selection == "both":
            return len(self.conv_ids) * 2
        else:
            return len(self.conv_ids)

    def __getitem__(self, i: int) -> ConversationBatch:
        # Determine whether the first or second speaker is designated the primary
        primary_speaker_selection_method: str = self.primary_speaker_selection
        if primary_speaker_selection_method == "both":
            if i < len(self.conv_ids):
                primary_speaker_selection_method = "first"
            else:
                i %= len(self.conv_ids)
                primary_speaker_selection_method = "second"

        # Load the conversation ID
        conv_id: Final[int] = self.conv_ids[i]

        # Load the conversation data
        conv_df = pd.read_csv(path.join(self.dataset_dir, "csv", f"{conv_id}.csv"))

        # Retrieve the transcript
        text: list[str] = conv_df.text.tolist()

        speaker_ids = torch.from_numpy(conv_df.spk_id.values.copy())
        # Assign labels of 1 and 2 for primary and secondary speaker, respectively,
        # according to the selection criteria
        speaker_side: Tensor = (
            torch.where(speaker_ids == speaker_ids[0], 1, 2)
            if primary_speaker_selection_method == "first"
            else torch.where(speaker_ids != speaker_ids[0], 1, 2)
        )

        # Load embeddings, if necessary
        segment_embeddings: Tensor | None = None
        if self.embeddings == "roberta":
            segment_embeddings = torch.load(
                path.join(self.dataset_dir, "roberta", f"{conv_id}.pt"),
                weights_only=True,
            )

        # Retrieve data from the loaded conversation
        features = torch.from_numpy(conv_df[self.features].values.astype("float32"))
        features_diff = torch.from_numpy(
            conv_df[[f"{x}_diff" for x in self.features]].values.astype("float32")
        )

        # Compute the side exchange vector
        speaker_exchange = F.one_hot(
            (
                speaker_side.diff(
                    dim=0,
                    prepend=torch.zeros(
                        1,
                    ),
                )
                != 0
            ).type(torch.long)
        ).type(torch.float)

        sides_exchange = {
            1: speaker_exchange[speaker_side == 1].unsqueeze(0),
            2: speaker_exchange[speaker_side == 2].unsqueeze(0),
        }

        # Compute side-specific deltas
        features_d_sides_all = features.clone()
        for side in [1, 2]:
            fd_side = features_d_sides_all[speaker_side == side]
            features_d_sides_all[speaker_side == side] = (
                fd_side - fd_side.mean(0)
            ) / fd_side.std(0)

        features_d_sides_all = features_d_sides_all.diff(
            dim=0, prepend=torch.zeros(1, len(self.features))
        )
        features_d_sides = {
            1: features_d_sides_all[speaker_side == 1].unsqueeze(0),
            2: features_d_sides_all[speaker_side == 2].unsqueeze(0),
        }

        # Compute side-specific embeddings
        embeddings_sides = {}
        if segment_embeddings is not None:
            embeddings_sides[1] = segment_embeddings[speaker_side == 1]
            embeddings_sides[2] = segment_embeddings[speaker_side == 2]

        # Normalization
        features_original = features.clone()
        if self.normalization == "zscore":
            if self.norm_z_mean is None or self.norm_z_std is None:
                raise Exception(
                    "If normalization method is zscore, norm_z_mean and norm_z_std must be given!"
                )
            features = (features - self.norm_z_mean["features"]) / self.norm_z_std[
                "features"
            ]
            features_diff = (
                features_diff - self.norm_z_mean["features_diff"]
            ) / self.norm_z_std["features_diff"]
        elif self.normalization == "zscore_conv":
            features = (features - features.mean(0)) / features.std(0)
            features_diff = (features_diff - features_diff.mean(0)) / features_diff.std(
                0
            )
        elif self.normalization == "zscore_conv_speaker":
            for speaker_id in speaker_ids.unique():
                speaker_id_mask = speaker_ids == speaker_id
                f = features[speaker_id_mask]
                features[speaker_id_mask] = (f - f.mean(0)) / f.std(0)

                f = features_diff[speaker_id_mask]
                features_diff[speaker_id_mask] = (f - f.mean(0)) / f.std(0)

        else:
            raise Exception(f"Unknown normalization method {self.normalization}")

        speaker_sex = torch.zeros((len(features), 2), dtype=torch.long)
        speaker_sex[(conv_df.sex == "m").values.copy(), 0] = 1
        speaker_sex[(conv_df.sex == "f").values.copy(), 1] = 1

        # Add a leading 0 to all data if requested
        if self.zero_pad:
            text.insert(0, "")
            features = F.pad(features, (0, 0, 1, 0))
            features_original = F.pad(features_original, (0, 0, 1, 0))
            features_diff = F.pad(features_diff, (0, 0, 1, 0))
            speaker_ids = F.pad(speaker_ids, (1, 0))
            speaker_side = F.pad(speaker_side, (1, 0))
            speaker_sex = F.pad(speaker_sex, (0, 0, 1, 0))
            speaker_exchange = F.pad(speaker_exchange, (0, 0, 1, 0))

            if segment_embeddings is not None:
                segment_embeddings = F.pad(segment_embeddings, (0, 0, 1, 0))

        return ConversationBatch(
            conv_ids=[conv_id],
            features=features.unsqueeze(0),
            features_original=features_original.unsqueeze(0),
            features_d=features_diff.unsqueeze(0),
            conv_lengths=torch.tensor([len(features)]),
            speaker_ids=speaker_ids.unsqueeze(0),
            speaker_sex=speaker_sex.unsqueeze(0),
            speaker_side=speaker_side.unsqueeze(0),
            sides_exchange=sides_exchange,
            segment_embeddings=segment_embeddings,
            features_sides={
                1: features[speaker_side == 1].unsqueeze(0),
                2: features[speaker_side == 2].unsqueeze(0),
            },
            embeddings_sides=embeddings_sides,
            features_d_sides=features_d_sides,
            sides_lengths={
                1: torch.tensor([(speaker_side == 1).sum()]),
                2: torch.tensor([(speaker_side == 2).sum()]),
            },
            text=[text],
            speaker_exchange=speaker_exchange.unsqueeze(0),
        )
