# main.py
from lightning.pytorch.cli import LightningCLI

from cdmodel.data import ConversationDataModule
from cdmodel.model import CDModel


def cli_main():
    cli = LightningCLI(CDModel, ConversationDataModule)


if __name__ == "__main__":
    cli_main()
