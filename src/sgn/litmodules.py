import pytorch_lightning as pl

from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torchmetrics import Accuracy

from sgn import SGN


class LitSGN(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        length=20,
        num_joints=25,
        num_features=3,
        lr=0.001,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sgn = SGN(num_classes, length, num_joints, num_features)

        self.metrics = nn.ModuleDict(
            {
                "train_acc": Accuracy(),
                "val_acc": Accuracy(),
                "test_acc": Accuracy(),
            }
        )

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.sgn(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.metrics["train_acc"](preds, y.int())
        self.log(
            "train_acc",
            self.metrics["train_acc"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.sgn(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        self.metrics["val_acc"](preds, y.int())
        self.log(
            "val_acc",
            self.metrics["val_acc"],
            on_step=True,
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.sgn(x)

        logits = logits.view((-1, x.size(0) // y.size(0), logits.size(1)))
        logits = logits.mean(1)

        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )

        self.metrics["test_acc"](preds, y.int())
        self.log("test_acc", self.metrics["test_acc"], on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": MultiStepLR(optimizer, milestones=[60, 90, 110], gamma=0.1),
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitSGN")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
