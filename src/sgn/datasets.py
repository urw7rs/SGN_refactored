from torch.utils.data import Dataset

import h5py

import numpy as np


class NTUDataset(Dataset):
    def __init__(self, h5_path, split):
        with h5py.File(h5_path, "r") as f:
            if split == "train":
                self.train_X = f["x"][:]
                self.train_Y = np.argmax(f["y"][:], -1)

                self.val_X = f["valid_x"][:]
                self.val_Y = np.argmax(f["valid_y"][:], -1)

                ## Combine the training data and validation data togehter as ST-GCN
                x = np.concatenate([self.train_X, self.val_X], axis=0)
                y = np.concatenate([self.train_Y, self.val_Y], axis=0)
            elif split in ["val", "test"]:
                x = f["test_x"][:]
                y = np.argmax(f["test_y"][:], -1)

        self.x = x
        self.y = np.array(y, dtype=int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
