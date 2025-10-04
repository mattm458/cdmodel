from os import path
from typing import Final, Literal, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from cdmodel.common.data import ConversationBatch

PrimarySpeakerSelectionStrategy = Literal["first"] | Literal["second"] | Literal["both"]


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        features: list[str],
        conv_ids: list[int],
        zero_pad: bool,
        embeddings: str | None,
        normalization: str,
        primary_speaker_selection: PrimarySpeakerSelectionStrategy,
        norm_z_mean: Optional[Tensor] = None,
        norm_z_std: Optional[Tensor] = None,
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

        self.norm_z_mean: Final[Tensor | None] = norm_z_mean
        self.norm_z_std: Final[Tensor | None] = norm_z_std

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

        # Load embeddings, if necessary
        segment_embeddings: Tensor | None = None
        if self.embeddings == "roberta":
            segment_embeddings = torch.load(
                path.join(self.dataset_dir, "roberta", f"{conv_id}.pt"),
                weights_only=True,
            )

        # Retrieve data from the loaded conversation
        features = torch.from_numpy(conv_df[self.features].values.astype("float32"))
        speaker_ids = torch.from_numpy(conv_df.spk_id.values)

        # Normalization
        if self.normalization == "zscore":
            if self.norm_z_mean is None or self.norm_z_std is None:
                raise Exception(
                    "If normalization method is zscore, norm_z_mean and norm_z_std must be given!"
                )
            features = (features - self.norm_z_mean) / self.norm_z_std
        elif self.normalization == "zscore_conv":
            features = (features - features.mean(0)) / features.std(0)
        elif self.normalization == "zscore_conv_speaker":
            for speaker_id in speaker_ids.unique():
                speaker_id_mask = speaker_ids == speaker_id
                f = features[speaker_id_mask]
                features[speaker_id_mask] = (f - f.mean(0)) / f.std(0)
        else:
            raise Exception(f"Unknown normalization method {self.normalization}")

        # Assign labels of 1 and 2 for primary and secondary speaker, respectively,
        # according to the selection criteria
        speaker_side: Tensor = (
            torch.where(speaker_ids == speaker_ids[0], 1, 2)
            if primary_speaker_selection_method == "first"
            else torch.where(speaker_ids != speaker_ids[0], 1, 2)
        )

        # Feature deltas are computed as the difference between features at each timestep.
        features_d = features.diff(dim=0, prepend=torch.zeros(1, len(self.features)))

        # Computing feature deltas at turn exchanges
        features_d_exchanges = features.clone()
        # Normalize both sides of the conversation
        for speaker_id in speaker_ids.unique():
            speaker_id_mask = speaker_ids == speaker_id
            f = features_d_exchanges[speaker_id_mask]
            features_d_exchanges[speaker_id_mask] = (f - f.mean(0)) / f.std(0)
        # Compute the delta
        features_d_exchanges = features_d.diff(dim=0)
        speaker_side_exc = speaker_side.diff() != 0
        features_d_sides_exchanges = {
            1: features_d_exchanges[
                ((speaker_side[1:] == 1) & speaker_side_exc)
            ].unsqueeze(0),
            2: features_d_exchanges[
                ((speaker_side[1:] == 2) & speaker_side_exc)
            ].unsqueeze(0),
        }

        # Add a leading 0 to all data if requested
        if self.zero_pad:
            features = F.pad(features, (0, 0, 1, 0))
            features_d = F.pad(features_d, (0, 0, 1, 0))
            speaker_ids = F.pad(speaker_ids, (1, 0))
            speaker_side = F.pad(speaker_side, (1, 0))

            if segment_embeddings is not None:
                segment_embeddings = F.pad(segment_embeddings, (0, 0, 1, 0))

        return ConversationBatch(
            conv_ids=[conv_id],
            features=features.unsqueeze(0),
            features_d=features_d.unsqueeze(0),
            conv_lengths=torch.tensor([len(features)]),
            speaker_ids=speaker_ids.unsqueeze(0),
            speaker_side=speaker_side.unsqueeze(0),
            segment_embeddings=segment_embeddings,
            features_sides={
                1: features[speaker_side == 1].unsqueeze(0),
                2: features[speaker_side == 2].unsqueeze(0),
            },
            features_d_sides={
                1: features_d[speaker_side == 1].unsqueeze(0),
                2: features_d[speaker_side == 2].unsqueeze(0),
            },
            sides_lengths={
                1: torch.tensor([(speaker_side == 1).sum()]),
                2: torch.tensor([(speaker_side == 2).sum()]),
            },
            features_d_sides_exchanges=features_d_sides_exchanges,
            features_d_sides_exchanges_lengths={
                1: torch.tensor([features_d_sides_exchanges[1].shape[1]]),
                2: torch.tensor([features_d_sides_exchanges[2].shape[1]]),
            },
        )
