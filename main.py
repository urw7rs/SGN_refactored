from pytorch_lightning.cli import LightningCLI

from sgn.litmodules import LitSGN
from sgn.datamodules import SGNDataModule


def main():
    cli = LightningCLI(
        LitSGN,
        SGNDataModule,
        save_config_overwrite=True,
    )


if __name__ == "__main__":
    main()
