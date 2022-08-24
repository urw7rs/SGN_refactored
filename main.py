from pytorch_lightning.cli import LightningCLI

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

from pytorch_lightning.loggers import WandbLogger


from sgn.litmodules import LitSGN
from sgn.datamodules import SGNDataModule


def main():
    wandb_logger = WandbLogger(
        name="SGN",
        project="Skeleton-Based Action Recognition",
        save_dir=".",
        log_model="all",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="max",
        save_last=True,
        dirpath=None,
        filename="{epoch}-{val_acc:.6f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    cli = LightningCLI(
        LitSGN,
        SGNDataModule,
        trainer_defaults=dict(
            callbacks=[
                RichModelSummary(),
                RichProgressBar(),
                checkpoint_callback,
                lr_monitor,
            ],
            logger=wandb_logger,
            max_epochs=120,
        ),
        save_config_overwrite=True,
    )


if __name__ == "__main__":
    main()
