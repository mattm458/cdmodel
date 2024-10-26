import json
from os import path
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import Dataset

from cdmodel.common import ConversationData


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        conv_ids: list[int],
        segment_features: list[str],
        zero_pad: bool = False,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.conv_ids: Final[list[int]] = conv_ids
        self.segment_features: Final[list[str]] = segment_features
        self.zero_pad: Final[bool] = zero_pad

    def __len__(self) -> int:
        return len(self.conv_ids)

    def __getitem__(self, i: int) -> ConversationData:
        conv_id: Final[int] = self.conv_ids[i]

        with open(path.join(self.dataset_dir, "segments", f"{conv_id}.json")) as infile:
            conv_data: Final[dict] = json.load(infile)

        segment_features: Tensor = torch.tensor(
            [conv_data[feature] for feature in self.segment_features]
        ).swapaxes(0, 1)

        embeddings: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-embeddings.pt")
        )

        embeddings_turn_len: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-lengths.pt")
        )

        return ConversationData(
            conv_id=[conv_id],
            segment_features=segment_features,
            embeddings=embeddings,
            embeddings_segment_len=embeddings_turn_len,
            num_segments=torch.tensor([len(segment_features)]),
        )
