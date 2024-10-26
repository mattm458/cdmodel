# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from cdmodel.model import ExampleModel
from cdmodel.data import ExampleDataModule


def cli_main():
    cli = LightningCLI(ExampleModel, ExampleDataModule)


if __name__ == "__main__":
    cli_main()
