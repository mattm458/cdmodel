from os import path
from typing import Final, Literal

import orjson
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from cdmodel.common.data import ConversationBatch

SpeakerSelectionStrategy = Literal["first"] | Literal["second"] | Literal["both"]


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        feature_names: list[str],
        conv_ids: list[int],
        zero_pad: bool,
        embeddings: str | None,
        normalization: str,
        speaker_1: SpeakerSelectionStrategy,
    ):
        self.dataset_dir: Final[str] = dataset_dir
        self.feature_names: list[str] = feature_names
        self.conv_ids: Final[list[int]] = conv_ids
        self.zero_pad: Final[bool] = zero_pad
        self.embeddings: Final[str | None] = embeddings
        self.normalization: Final[str] = normalization
        self.speaker_1: Final[SpeakerSelectionStrategy] = speaker_1

    def __len__(self) -> int:
        if self.speaker_1 == "both":
            return len(self.conv_ids) * 2
        else:
            return len(self.conv_ids)

    def __getitem__(self, i: int) -> ConversationBatch:
        # Determine whether the first or second speaker is designated the primary
        primary_speaker_selection_method: str = self.speaker_1
        if primary_speaker_selection_method == "both":
            if i < len(self.conv_ids):
                primary_speaker_selection_method = "first"
            else:
                i %= len(self.conv_ids)
                primary_speaker_selection_method = "second"

        # Load the conversation ID
        conv_id: Final[int] = self.conv_ids[i]

        # Load the conversation data
        with open(path.join(self.dataset_dir, "segments", f"{conv_id}.json")) as infile:
            conv_data: Final[dict] = orjson.loads(infile.read())

        # Retrieve data from the loaded conversation
        features: Tensor = torch.tensor([conv_data[f] for f in self.feature_names]).T
        speaker_ids: Tensor = torch.tensor(conv_data["speaker_id"])

        for speaker_id in speaker_ids.unique():
            speaker_id_mask = speaker_ids == speaker_id
            f = features[speaker_id_mask]
            features[speaker_id_mask] = (f - f.mean(0)) / f.std(0)

        # Assign labels of 1 and 2 for primary and secondary speaker, respectively,
        # according to the selection criteria
        speaker_side: Tensor = (
            torch.where(speaker_ids == speaker_ids[0], 1, 2)
            if primary_speaker_selection_method == "first"
            else torch.where(speaker_ids != speaker_ids[0], 1, 2)
        )

        # Add a leading 0 to all data if requested
        if self.zero_pad:
            features = F.pad(features, (0, 0, 1, 0))
            speaker_ids = F.pad(speaker_ids, (1, 0))
            speaker_side = F.pad(speaker_side, (1, 0))

        return ConversationBatch(
            features=features.unsqueeze(0),
            conv_lengths=torch.tensor([len(features)]),
            speaker_ids=speaker_ids.unsqueeze(0),
            speaker_designation=speaker_side.unsqueeze(0),
        )
