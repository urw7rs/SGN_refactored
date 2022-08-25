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

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.metrics["train_acc"](preds, y.int())
        self.log(
            "train_acc",
            self.metrics["train_acc"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.sgn(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.metrics["val_acc"](preds, y.int())
        self.log(
            "val_acc",
            self.metrics["val_acc"],
            prog_bar=True,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.sgn(x)

        logits = logits.view(y.size(0), -1, self.hparams.num_classes)
        logits = logits.mean(1)

        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log("test_loss", loss, on_epoch=True)

        self.metrics["test_acc"](preds, y.int())
        self.log("test_acc", self.metrics["test_acc"], on_epoch=True)

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


    def on_fit_start(self) -> None:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        for batch in self.trainer.datamodule.train_dataloader():
            break

        x = batch[0].to(self.device)
        x = x[:1]

        flops = FlopCountAnalysis(self.sgn, x)
        print(flop_count_table(flops))
