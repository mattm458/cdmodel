# main.py
import torch
from lightning.pytorch.cli import LightningCLI

from cdmodel.data import ConversationDataModule
from cdmodel.model import CDModel


def cli_main():
    torch.set_float32_matmul_precision("high")
    cli = LightningCLI(CDModel, ConversationDataModule)


if __name__ == "__main__":
    cli_main()
