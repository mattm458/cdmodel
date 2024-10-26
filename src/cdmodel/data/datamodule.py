from os import path
from typing import Final

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cdmodel.data.collate_fn import collate_fn
from cdmodel.data.dataset import ConversationDataset


def __load_set_ids(dataset_dir: str, dataset_subset: str, set: str) -> list[int]:
    with open(path.join(dataset_dir, f"{set}-{dataset_subset}.csv")) as infile:
        return [int(x) for x in infile.readlines() if len(x) > 0]


class ConversationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        data_subset: str,
        segment_features: list[str],
        zero_pad: bool,
        batch_size: int,
        num_workers: int = 4,
    ):
        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.data_subset: Final[str] = data_subset
        self.segment_features: Final[list[str]] = segment_features
        self.zero_pad: Final[bool] = zero_pad
        self.batch_size: Final[int] = batch_size
        self.num_workers: Final[int] = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.dataset_train = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    conv_ids=__load_set_ids(
                        self.dataset_dir, self.data_subset, "train"
                    ),
                    segment_features=self.segment_features,
                    zero_pad=self.zero_pad,
                )
                self.dataset_validate = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    conv_ids=__load_set_ids(self.dataset_dir, self.data_subset, "val"),
                    segment_features=self.segment_features,
                    zero_pad=self.zero_pad,
                )
            case "validate":
                self.dataset_validate = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    conv_ids=__load_set_ids(self.dataset_dir, self.data_subset, "val"),
                    segment_features=self.segment_features,
                    zero_pad=self.zero_pad,
                )
            case "test":
                self.dataset_test = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    conv_ids=__load_set_ids(self.dataset_dir, self.data_subset, "test"),
                    segment_features=self.segment_features,
                    zero_pad=self.zero_pad,
                )
            case "predict":
                self.dataset_predict = ConversationDataset(
                    dataset_dir=self.dataset_dir,
                    conv_ids=__load_set_ids(self.dataset_dir, self.data_subset, "test"),
                    segment_features=self.segment_features,
                    zero_pad=self.zero_pad,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
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
