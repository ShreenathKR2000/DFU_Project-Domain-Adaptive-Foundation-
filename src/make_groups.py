"""
Find near-duplicate images and cluster them into groups, so that train / val /
test splits can keep each group together (``--group_csv`` in ``src.train``).

DFUC2021 ships no patient IDs, and a random split can put near-identical images
of one ulcer on both sides, which inflates validation/test scores (internal
0.88 macro-F1 vs ~0.55-0.65 on the hidden leaderboard test set). This script
(1) embeds every labeled image with the frozen DINOv2 backbone, (2) links pairs
with cosine similarity >= ``--threshold`` and takes connected components as
groups, and (3) prints a leak diagnostic: how many images in a random test split
have a near-duplicate in the random training split, vs. the group-aware split.

Usage
-----
    python -m src.make_groups                       # threshold 0.90
    python -m src.make_groups --threshold 0.85 --out groups.csv
Look at the printed threshold table first and pick one that groups obvious
duplicates without merging everything into one giant cluster.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from src.dataset import (
    DFUDataset,
    check_labeled,
    get_eval_transforms,
    load_and_split_csv,
    make_splits,
)
from src.common import CSV_PATH, IMG_DIR


@torch.no_grad()
def embed(df: pd.DataFrame, model_name: str, batch_size: int, num_workers: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dinov2Model.from_pretrained(model_name).to(device).eval()
    loader = DataLoader(
        DFUDataset(df, IMG_DIR, transform=get_eval_transforms()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    feats = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        cls = model(pixel_values=images.to(device)).last_hidden_state[:, 0]
        feats.append(torch.nn.functional.normalize(cls, dim=1).cpu())
    return torch.cat(feats).numpy()


def cluster(sim: np.ndarray, threshold: float) -> np.ndarray:
    adj = csr_matrix(sim >= threshold)
    _, labels = connected_components(adj, directed=False)
    return labels


def leak_rate(sim: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, thr: float) -> float:
    """Share of test images with a training neighbour at cosine >= thr."""
    nn_sim = sim[np.ix_(test_idx, train_idx)].max(axis=1)
    return float((nn_sim >= thr).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--model_name", default="facebook/dinov2-base")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="groups.csv")
    args = ap.parse_args()

    labeled_df, _ = load_and_split_csv(CSV_PATH)
    check_labeled(labeled_df)
    print(f"Embedding {len(labeled_df)} labeled images ...")
    feats = embed(labeled_df, args.model_name, args.batch_size, args.num_workers)
    sim = feats @ feats.T
    np.fill_diagonal(sim, -1.0)

    print("\nthreshold   groups   largest   images-in-multi-image-groups")
    for thr in (0.80, 0.85, 0.90, 0.95, 0.98):
        g = cluster(sim, thr)
        sizes = np.bincount(g)
        print(f"  {thr:.2f}     {len(sizes):>6}   {sizes.max():>7}   {int(sizes[sizes > 1].sum()):>8}")

    groups = cluster(sim, args.threshold)
    sizes = np.bincount(groups)
    print(f"\nUsing threshold {args.threshold}: {len(sizes)} groups, largest {sizes.max()}")
    pd.DataFrame({"image": labeled_df["image"], "group": groups}).to_csv(args.out, index=False)
    print(f"Groups → {args.out}")

    # Leak diagnostic: random split vs group-aware split (same seed).
    idx_of = {n: i for i, n in enumerate(labeled_df["image"])}
    rows = []
    for name, grp in (("random split", None), ("group-aware split", groups)):
        tr, _, te = make_splits(labeled_df, seed=args.seed, groups=grp)
        tr_i = np.array([idx_of[n] for n in tr["image"]])
        te_i = np.array([idx_of[n] for n in te["image"]])
        rows.append((name, *[leak_rate(sim, tr_i, te_i, t) for t in (0.90, 0.95, 0.98)]))
    print("\nShare of TEST images that have a TRAIN neighbour with cosine >= t")
    print(f"{'':<20}{'t=0.90':>8}{'t=0.95':>8}{'t=0.98':>8}")
    for name, *v in rows:
        print(f"{name:<20}" + "".join(f"{x:>8.1%}" for x in v))
    print("\nA large drop for the group-aware split confirms that random splits leak "
          "near-duplicates; use --group_csv for validation/model selection.")


if __name__ == "__main__":
    main()
