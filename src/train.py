"""
Training loop for DFU classification.

Usage
-----
Full run:
    python -m src.train

Debug run (tiny subset, batch_size=2, 2 epochs — fits GTX 1650 4 GB):
    python -m src.train --debug
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_metric_learning.losses import SupConLoss

from src.dataset import (
    DFUDataset,
    get_train_transforms,
    get_eval_transforms,
    load_and_split_csv,
)
from src.model import DFUDinoLoRA

# ── paths (relative to project root) ────────────────────────────────────────

DATA_DIR = os.path.join("Data", "DFUC2021_train")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
CONTRASTIVE_WEIGHT = 0.5  # λ for SupConLoss (CE weight is 1 − λ)

DEBUG_SAMPLES = 20
DEBUG_BATCH_SIZE = 2
DEBUG_EPOCHS = 2


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DFU DINOv2-LoRA model")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a tiny subset (20 images, batch_size=2) to verify the pipeline.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Path to Phase-1 LoRA checkpoint (e.g. checkpoints/dfu_pretrained_backbone.pt).",
    )
    return parser.parse_args()


def build_loader(
    df: pd.DataFrame,
    img_dir: str,
    transform,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = DFUDataset(df, img_dir, transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )


# ── training step ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.Module,
    con_criterion: SupConLoss,
    device: torch.device,
    contrastive_weight: float = CONTRASTIVE_WEIGHT,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_con = 0.0
    correct = 0
    n_samples = 0

    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        logits, embeddings = model(images)

        # Cross-entropy loss
        ce_loss = ce_criterion(logits, labels)

        # Supervised contrastive loss on L2-normalised embeddings
        emb_norm = F.normalize(embeddings, dim=1)
        con_loss = con_criterion(emb_norm, labels)

        loss = (1 - contrastive_weight) * ce_loss + contrastive_weight * con_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_ce += ce_loss.item() * images.size(0)
        total_con += con_loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n_samples += images.size(0)

    return {
        "loss": total_loss / n_samples,
        "ce_loss": total_ce / n_samples,
        "con_loss": total_con / n_samples,
        "accuracy": correct / n_samples,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    debug = args.debug
    epochs = args.epochs or (DEBUG_EPOCHS if debug else DEFAULT_EPOCHS)
    batch_size = args.batch_size or (DEBUG_BATCH_SIZE if debug else DEFAULT_BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Debug  : {debug}")
    print(f"Epochs : {epochs}")
    print(f"Batch  : {batch_size}")
    print(f"LoRA   : {args.pretrained_lora_path or 'scratch'}")

    # ── data ─────────────────────────────────────────────────────────────
    labeled_df, _ = load_and_split_csv(CSV_PATH)

    if debug:
        labeled_df = labeled_df.head(DEBUG_SAMPLES)

    train_loader = build_loader(
        labeled_df, IMG_DIR, get_train_transforms(), batch_size
    )

    # ── model ────────────────────────────────────────────────────────────
    model = DFUDinoLoRA(pretrained_lora_path=args.pretrained_lora_path)
    model.to(device)
    model.print_trainable_parameters()

    # ── optimiser & loss ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    ce_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss()

    # ── training loop ────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, con_criterion, device
        )
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"ce={metrics['ce_loss']:.4f}  "
            f"con={metrics['con_loss']:.4f}  "
            f"acc={metrics['accuracy']:.3f}  "
            f"({elapsed:.1f}s)"
        )

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", "dfu_dino_lora.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
