"""Paths, seeding and the DataLoader factory shared by all entry points."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DFUDataset

# Paths (relative to the project root); override the data location with DFU_DATA_DIR
DATA_DIR = os.environ.get("DFU_DATA_DIR", os.path.join("Data", "DFUC2021_train"))
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")


def enable_tf32() -> None:
    """TF32 matmuls on Ampere/Ada GPUs: much faster fp32, negligible accuracy impact."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int) -> None:
    enable_tf32()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    df,
    img_dir: str,
    transform,
    batch_size: int,
    shuffle: bool = True,
    sampler=None,
    num_workers: int = 2,
) -> DataLoader:
    ds = DFUDataset(df, img_dir, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
