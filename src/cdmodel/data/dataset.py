import json
from os import path
from typing import Final

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cdmodel.common import ConversationData


def load_set_ids(dataset_dir: str, dataset_subset: str, set: str) -> list[int]:
    with open(path.join(dataset_dir, f"{set}-{dataset_subset}.csv")) as infile:
        return [int(x) for x in infile.readlines() if len(x) > 0]


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        segment_features: list[str],
        zero_pad: bool,
        subset: str,
        set: str,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.conv_ids: Final[list[int]] = load_set_ids(
            dataset_dir=dataset_dir,
            dataset_subset=subset,
            set=set,
        )
        self.segment_features: Final[list[str]] = segment_features
        self.zero_pad: Final[bool] = zero_pad
        self.speaker_ids: Final[dict[int, int]] = pd.read_csv(
            path.join(dataset_dir, f"speaker-ids-{subset}.csv"),
            index_col="speaker_id",
        )["idx"].to_dict()

    def __len__(self) -> int:
        return len(self.conv_ids)

    def __getitem__(self, i: int) -> ConversationData:
        conv_id: Final[int] = self.conv_ids[i]

        with open(path.join(self.dataset_dir, "segments", f"{conv_id}.json")) as infile:
            conv_data: Final[dict] = json.load(infile)

        segment_features: Tensor = (
            torch.tensor([conv_data[feature] for feature in self.segment_features])
            .swapaxes(0, 1)
            .unsqueeze(0)
        )
        segment_features_delta = segment_features.diff(
            dim=1, prepend=torch.zeros(1, 1, segment_features.shape[2])
        )

        embeddings: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-embeddings.pt"),
            weights_only=True,
        )

        embeddings_turn_len: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-lengths.pt"),
            weights_only=True,
        )

        speaker_id: list[list[int]] = [conv_data["speaker_id"]]
        speaker_id_idx: Tensor = torch.tensor(
            [self.speaker_ids[x] for x in conv_data["speaker_id"]],
            dtype=torch.long,
        ).unsqueeze(0)

        return ConversationData(
            conv_id=[conv_id],
            segment_features=segment_features,
            segment_features_delta=segment_features_delta,
            embeddings=embeddings,
            embeddings_segment_len=embeddings_turn_len,
            num_segments=[segment_features.shape[1]],
            speaker_id=speaker_id,
            speaker_id_idx=speaker_id_idx,
        )
