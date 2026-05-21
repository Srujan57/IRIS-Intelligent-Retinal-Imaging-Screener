"""
dataset.py

PyTorch Dataset class for loading preprocessed OCT scans.
Reads image paths and labels from data/labels.csv,
applies train/val/test splits from data/splits.csv,
and feeds samples into the DataLoader for training and evaluation.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset

ON_KAGGLE = os.path.exists("/kaggle/input")
LABELS_CSV = "/kaggle/working/data/labels.csv" if ON_KAGGLE else "data/labels.csv"


LABEL_MAP = {
    "healthy": 0,
    "eating_disorder_pattern": 1
}


class OCTDataset(Dataset):
    def __init__(self, labels_csv, split, transform=None):
        """
        labels_csv  : path to data/labels.csv
        split       : one of 'train', 'val', 'test'
        transform   : augmentation or validation transform to apply
        """
        df = pd.read_csv(labels_csv, comment="#")
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # load preprocessed tensor from disk
        image = torch.load(row["image_path"], weights_only=True)

        # encode label as integer
        label = LABEL_MAP[row["label"]]

        # apply transform if provided
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloader(labels_csv, split, transform, batch_size, shuffle=True):
    from torch.utils.data import DataLoader

    dataset = OCTDataset(
        labels_csv=labels_csv,
        split=split,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
