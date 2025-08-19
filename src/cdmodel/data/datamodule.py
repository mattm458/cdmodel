from os import path
from typing import Final

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from cdmodel.data.collate_fn import collate_fn
from torch.utils.data import DataLoader, random_split

from cdmodel.data.dataset import ConversationDataset, PrimarySpeakerSelectionStrategy


class ConversationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        features,
        zero_pad: bool,
        embeddings: str | None,
        normalization: str,
        primary_speaker_selection: PrimarySpeakerSelectionStrategy,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.batch_size: Final[int] = batch_size

        self.dataset = ConversationDataset(
            dataset_dir=dataset_dir,
            features=features,
            conv_ids=np.sort(
                pd.read_csv(path.join(dataset_dir, "data.csv"), engine="pyarrow")[
                    "id"
                ].unique()
            ).tolist(),
            zero_pad=zero_pad,
            embeddings=embeddings,
            normalization=normalization,
            primary_speaker_selection=primary_speaker_selection,
        )

    def prepare_data(self):
        print("Preparing data")

    def setup(self, stage: str):
        self.dataset_train, self.dataset_validate, self.dataset_test = random_split(
            self.dataset,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_validate,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
