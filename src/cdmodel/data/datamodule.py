from os import path
from typing import Final

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split

from cdmodel.data.collate_fn import collate_fn
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
        num_workers: int,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.batch_size: Final[int] = batch_size
        self.features: Final[list[str]] = features
        self.zero_pad: Final[bool] = zero_pad
        self.embeddings: Final[str | None] = embeddings
        self.normalization: Final[str] = normalization
        self.primary_speaker_selection: Final[PrimarySpeakerSelectionStrategy] = (
            primary_speaker_selection
        )
        self.num_workers: Final[int] = num_workers

    def prepare_data(self):
        print("Preparing data")

    def setup(self, stage: str):
        self.conv_ids_train, self.conv_ids_test = train_test_split(
            np.sort(
                pd.read_csv(path.join(self.dataset_dir, "data.csv"), engine="pyarrow")[
                    "id"
                ].unique()
            ).tolist(),
            random_state=42,
            train_size=0.8,
        )
        self.conv_ids_test, self.conv_ids_val = train_test_split(
            self.conv_ids_test, random_state=42, test_size=0.5
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            ConversationDataset(
                dataset_dir=self.dataset_dir,
                features=self.features,
                conv_ids=self.conv_ids_train,
                zero_pad=self.zero_pad,
                embeddings=self.embeddings,
                normalization=self.normalization,
                primary_speaker_selection=self.primary_speaker_selection,
            ),
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ConversationDataset(
                dataset_dir=self.dataset_dir,
                features=self.features,
                conv_ids=self.conv_ids_val,
                zero_pad=self.zero_pad,
                embeddings=self.embeddings,
                normalization=self.normalization,
                primary_speaker_selection=self.primary_speaker_selection,
            ),
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            ConversationDataset(
                dataset_dir=self.dataset_dir,
                features=self.features,
                conv_ids=self.conv_ids_test,
                zero_pad=self.zero_pad,
                embeddings=self.embeddings,
                normalization=self.normalization,
                primary_speaker_selection=self.primary_speaker_selection,
            ),
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
