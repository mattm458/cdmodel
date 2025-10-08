from os import path
from typing import Final, Literal

import pandas as pd
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from cdmodel.data.collate_fn import collate_fn
from cdmodel.data.dataset import (
    ConversationDataset,
    NormalizationStrategy,
    PrimarySpeakerSelectionStrategy,
)
from cdmodel.util.split import split_by_id, split_disjoint

SplitStrategy = Literal["id"] | Literal["disjoint"]


class ConversationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        split_strategy: SplitStrategy,
        batch_size: int,
        features,
        zero_pad: bool,
        embeddings: str | None,
        normalization: NormalizationStrategy,
        primary_speaker_selection: PrimarySpeakerSelectionStrategy,
        num_workers: int,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.split_strategy: Final[SplitStrategy] = split_strategy
        self.batch_size: Final[int] = batch_size
        self.features: Final[list[str]] = features
        self.zero_pad: Final[bool] = zero_pad
        self.embeddings: Final[str | None] = embeddings
        self.normalization: Final[str] = normalization
        self.primary_speaker_selection: Final[PrimarySpeakerSelectionStrategy] = (
            primary_speaker_selection
        )
        self.num_workers: Final[int] = num_workers

        self.norm_z_mean: dict[str, Tensor] = {}
        self.norm_z_std: dict[str, Tensor] = {}

    def prepare_data(self):
        print("Preparing data")

    def setup(self, stage: str):
        df = pd.read_csv(path.join(self.dataset_dir, "data_all.csv"), engine="pyarrow")

        self.conv_ids_train: list[int]
        self.conv_ids_val: list[int]
        self.conv_ids_test: list[int]
        if self.split_strategy == "id":
            self.conv_ids_train, self.conv_ids_val, self.conv_ids_test = split_by_id(
                set(df["conv_id"].unique().tolist())
            )
        elif self.split_strategy == "disjoint":
            self.conv_ids_train, self.conv_ids_val, self.conv_ids_test = split_disjoint(
                df
            )
        else:
            raise Exception(f"Unknown split strategy {self.split_strategy}")

        for f in ["", "_diff", "_spk_diff"]:
            self.norm_z_mean[f"features{f}"] = torch.from_numpy(
                df.loc[
                    df.conv_id.isin(self.conv_ids_train),
                    [f"{x}{f}" for x in self.features],
                ].values.astype("float32")
            ).mean(0)
            self.norm_z_std[f"features{f}"] = torch.from_numpy(
                df.loc[
                    df.conv_id.isin(self.conv_ids_train), self.features
                ].values.astype("float32")
            ).std(0)

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
                norm_z_mean=self.norm_z_mean,
                norm_z_std=self.norm_z_std,
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
                norm_z_mean=self.norm_z_mean,
                norm_z_std=self.norm_z_std,
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
                norm_z_mean=self.norm_z_mean,
                norm_z_std=self.norm_z_std,
            ),
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
