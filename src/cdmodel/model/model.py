from lightning import pytorch as pl
from torch import Tensor


class ExampleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx: int):
        pass

    def validation_step(self, batch, batch_idx: int):
        pass

    def test_step(self, batch, batch_idx: int):
        pass
