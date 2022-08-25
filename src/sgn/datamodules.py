import pytorch_lightning as pl

from typing import Optional

from sgn.compat.data import NTUDataLoaders


class SGNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str = "NTU",
        cross: str = "sub",
        no_aug: bool = False,
        seq_len: int = 20,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        if cross == "sub":
            case = 0
        elif cross == "view":
            case = 1

        self.loader_kwargs = dict(dataset=name, case=case, aug=not no_aug, seg=seq_len)

    def setup(self, stage: Optional[str] = None):
        self.ntu_dataloaders = NTUDataLoaders(**self.loader_kwargs)

    def train_dataloader(self):
        return self.ntu_dataloaders.get_train_loader(self.batch_size, self.num_workers)

    def val_dataloader(self):
        return self.ntu_dataloaders.get_val_loader(self.batch_size, self.num_workers)

    def test_dataloader(self):
        return self.ntu_dataloaders.get_test_loader(self.batch_size, self.num_workers)
