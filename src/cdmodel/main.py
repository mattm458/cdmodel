from lightning.pytorch.cli import LightningCLI
import torch

# simple demo classes for your convenience
from cdmodel.model import CDModel
from cdmodel.data import ConversationDataModule


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(CDModel, ConversationDataModule)


if __name__ == "__main__":
    cli_main()
