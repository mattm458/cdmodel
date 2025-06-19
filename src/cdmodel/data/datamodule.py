from os import path
from typing import Final

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cdmodel.common.role_assignment import RoleAssignmentStrategy, PredictionType
from cdmodel.data.collate_fn import collate_fn
from cdmodel.data.dataset import ConversationDataset


def load_set_ids(dataset_dir: str, dataset_subset: str, set: str) -> list[int]:
    with open(path.join(dataset_dir, f"{set}-{dataset_subset}.csv")) as infile:
        return [x.strip() for x in infile.readlines() if len(x) > 0]


class ConversationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        data_subset: str,
        segment_features: list[str],
        zero_pad: bool,
        batch_size: int,
        num_workers: int,
        role_type: str,
        role_assignment_strategy: str,
        embeddings_type: str | None,
        shuffle_training: bool = True,
        drop_last_training: bool = True,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.data_subset: Final[str] = data_subset
        self.segment_features: Final[list[str]] = segment_features
        self.zero_pad: Final[bool] = zero_pad
        self.batch_size: Final[int] = batch_size
        self.num_workers: Final[int] = num_workers
        self.role_type: Final[PredictionType] = PredictionType[role_type]
        self.embeddings_type: Final[str | None] = embeddings_type
        self.shuffle_training: Final[bool] = shuffle_training
        self.drop_last_training: Final[bool] = drop_last_training

        if role_assignment_strategy == "random":
            raise NotImplementedError(
                "'random' role assignment strategy has been removed in favor of 'both'"
            )

        self.role_assignment_strategy: Final[RoleAssignmentStrategy] = (
            RoleAssignmentStrategy[role_assignment_strategy]
        )
        self.speaker_ids: Final[dict[int, int]] = pd.read_csv(
            path.join(self.dataset_dir, f"speaker-ids-{self.data_subset}.csv"),
            index_col="speaker_id",
        )["idx"].to_dict()

    def prepare_data(self) -> None:
        return

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.dataset_train = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    feature_names=self.segment_features,
                    zero_pad=self.zero_pad,
                    role_type=self.role_type,
                    role_assignment_strategy=self.role_assignment_strategy,
                    conv_ids=load_set_ids(
                        dataset_dir=self.dataset_dir,
                        dataset_subset=self.data_subset,
                        set="train",
                    ),
                    speaker_ids=self.speaker_ids,
                    embeddings_type=self.embeddings_type,
                )
                self.dataset_validate = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    feature_names=self.segment_features,
                    zero_pad=self.zero_pad,
                    role_type=self.role_type,
                    role_assignment_strategy=self.role_assignment_strategy,
                    conv_ids=load_set_ids(
                        dataset_dir=self.dataset_dir,
                        dataset_subset=self.data_subset,
                        set="val",
                    ),
                    speaker_ids=self.speaker_ids,
                    embeddings_type=self.embeddings_type,
                )
            case "validate":
                self.dataset_validate = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    feature_names=self.segment_features,
                    zero_pad=self.zero_pad,
                    role_type=self.role_type,
                    role_assignment_strategy=self.role_assignment_strategy,
                    conv_ids=load_set_ids(
                        dataset_dir=self.dataset_dir,
                        dataset_subset=self.data_subset,
                        set="val",
                    ),
                    speaker_ids=self.speaker_ids,
                    embeddings_type=self.embeddings_type,
                )
            case "test":
                self.dataset_test = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    feature_names=self.segment_features,
                    zero_pad=self.zero_pad,
                    role_type=self.role_type,
                    role_assignment_strategy=self.role_assignment_strategy,
                    conv_ids=load_set_ids(
                        dataset_dir=self.dataset_dir,
                        dataset_subset=self.data_subset,
                        set="test",
                    ),
                    speaker_ids=self.speaker_ids,
                    embeddings_type=self.embeddings_type,
                )
            case "predict":
                self.dataset_predict = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    feature_names=self.segment_features,
                    zero_pad=self.zero_pad,
                    role_type=self.role_type,
                    role_assignment_strategy=self.role_assignment_strategy,
                    conv_ids=load_set_ids(
                        dataset_dir=self.dataset_dir,
                        dataset_subset=self.data_subset,
                        set="test",
                    ),
                    speaker_ids=self.speaker_ids,
                    embeddings_type=self.embeddings_type,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training,
            drop_last=self.drop_last_training,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_validate,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_predict,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
