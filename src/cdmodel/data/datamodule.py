from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cdmodel.data.dataset import ExampleDataset


class ExampleDataModule(LightningDataModule):
    def __init__(self, data_dir: str):
        super().__init__()

        self.data_dir = data_dir

    def prepare_data(self):
        print("Preparing data")

    def setup(self, stage: str):
        match stage:
            case "fit":
                self.dataset_train = ExampleDataset(self.data_dir)
                self.dataset_validate = ExampleDataset(self.data_dir)
            case "validate":
                self.dataset_validate = ExampleDataset(self.data_dir)
            case "test":
                self.dataset_test = ExampleDataset(self.data_dir)
            case "predict":
                self.dataset_predict = ExampleDataset(self.data_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, batch_size=2)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_validate, batch_size=2)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_test, batch_size=2)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_predict, batch_size=2)
