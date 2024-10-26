# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from project.model import ExampleModel
from project.data import ExampleDataModule


def cli_main():
    cli = LightningCLI(ExampleModel, ExampleDataModule)


if __name__ == "__main__":
    cli_main()
