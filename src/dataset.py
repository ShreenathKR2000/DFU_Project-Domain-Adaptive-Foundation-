"""
DFU Dataset — handles both labeled and unlabeled subsets.

The CSV schema (one-hot encoded):
    image, none, infection, ischaemia, both

Labeled rows have exactly one column set to 1.
Unlabeled rows have all four label columns empty/NaN.

Label mapping:
    0 → none
    1 → infection
    2 → ischaemia
    3 → both
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── constants ────────────────────────────────────────────────────────────────

LABEL_COLS = ["none", "infection", "ischaemia", "both"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── transforms ───────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── dataset ──────────────────────────────────────────────────────────────────

class DFUDataset(Dataset):
    """PyTorch Dataset for Diabetic Foot Ulcer images.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``image`` column with filenames.  If the label columns
        (``none``, ``infection``, ``ischaemia``, ``both``) are present **and**
        non-null for a given row, the sample is treated as labeled.
    img_dir : str
        Path to the directory that contains the image files.
    transform : optional torchvision transform
        Applied to each PIL image before returning.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform or get_eval_transforms()

        # Determine which rows carry labels.
        if all(c in self.df.columns for c in LABEL_COLS):
            self.has_labels = self.df[LABEL_COLS].notna().all(axis=1)
        else:
            self.has_labels = pd.Series(False, index=self.df.index)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.img_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Return (image, label) when labeled, else just image.
        if self.has_labels.iloc[idx]:
            label = int(row[LABEL_COLS].values.argmax())
            return image, label

        return image


# ── helper: split CSV into labeled / unlabeled frames ────────────────────────

def load_and_split_csv(
    csv_path: str, n_labeled: int = 5955
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(labeled_df, unlabeled_df)`` from the raw train.csv."""
    df = pd.read_csv(csv_path)
    labeled_df = df.iloc[:n_labeled].copy()
    unlabeled_df = df.iloc[n_labeled:].copy()
    return labeled_df, unlabeled_df


# ── helpers: labels and stratified train / val / test split ──────────────────

def labels_from_df(df: pd.DataFrame) -> np.ndarray:
    """Integer class index (0..3) for every row of a labeled frame."""
    return df[LABEL_COLS].values.argmax(axis=1)


def check_labeled(df: pd.DataFrame) -> None:
    """Fail early if the 'labeled' frame contains unlabeled rows."""
    vals = df[LABEL_COLS]
    if vals.isna().any().any():
        raise ValueError(
            "Labeled frame contains NaN label values; the CSV row order does "
            "not match the assumed 5955 labeled + 3994 unlabeled layout."
        )
    n_bad = int((vals.sum(axis=1) != 1).sum())
    if n_bad:
        print(f"WARNING: {n_bad} labeled rows are not exactly one-hot; "
              "argmax will be used for them.")


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    """Number of samples per class name."""
    counts = np.bincount(labels_from_df(df), minlength=len(LABEL_COLS))
    return {name: int(n) for name, n in zip(LABEL_COLS, counts)}


def make_splits(
    labeled_df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified ``(train, val, test)`` split of the labeled frame.

    The split depends only on ``seed``, so runs that share a seed (e.g. the
    scratch and Phase-1 arms of an experiment) see identical splits.  Falls
    back to an unstratified split when a class is too small to stratify
    (only happens on tiny debug subsets).
    """
    from sklearn.model_selection import train_test_split

    y = labels_from_df(labeled_df)
    idx = np.arange(len(labeled_df))

    def _split(ix, frac, yy):
        try:
            return train_test_split(ix, test_size=frac, stratify=yy, random_state=seed)
        except ValueError:
            return train_test_split(ix, test_size=frac, random_state=seed)

    trainval_idx, test_idx = _split(idx, test_frac, y)
    train_idx, val_idx = _split(
        trainval_idx, val_frac / (1.0 - test_frac), y[trainval_idx]
    )
    return tuple(  # type: ignore[return-value]
        labeled_df.iloc[ix].reset_index(drop=True)
        for ix in (train_idx, val_idx, test_idx)
    )
