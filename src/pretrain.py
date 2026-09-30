"""
Domain-Adaptive Pre-training — Masked Image Modeling (SimMIM-style).

Phase 1 of the two-stage pipeline:
    1. **Pre-train** LoRA adapters on *unlabeled* DFU images (this script).
    2. Fine-tune LoRA adapters + classification head on *labeled* images
       (``train.py``).

Uses ONLY the unlabeled portion of the DFUC2021 dataset (rows 5 955 → 9 949).
Randomly masks 60 % of input patches, reconstructs original pixel values
via a lightweight linear decoder, and trains with L1 loss.

Usage
-----
Debug run (20 images, batch_size=2, 2 epochs — fits GTX 1650 4 GB):
    python -m src.pretrain --debug

Full run:
    python -m src.pretrain

Override defaults:
    python -m src.pretrain --epochs 30 --batch_size 4 --lr 5e-5 --mask_ratio 0.5

After pre-training, load the adapted LoRA weights for Phase 2:
    model = DFUDinoLoRA()
    model.backbone.load_state_dict(
        torch.load("checkpoints/dfu_pretrained_backbone.pt")
    )
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common import CSV_PATH, IMG_DIR, build_loader
from src.dataset import get_train_transforms, load_and_split_csv
from src.pretrain_model import DFUDinoLoRAForMIM

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_EPOCHS     = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR         = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_MASK_RATIO = 0.6

DEBUG_SAMPLES    = 20
DEBUG_BATCH_SIZE = 2
DEBUG_EPOCHS     = 2


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SimMIM pre-training on unlabeled DFU data"
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Tiny subset (20 images, batch_size=2, 2 epochs) for quick sanity check.",
    )
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=DEFAULT_LR)
    p.add_argument("--mask_ratio", type=float, default=DEFAULT_MASK_RATIO)
    p.add_argument(
        "--extra_img_dirs", nargs="*", default=[],
        help="Extra directories of unlabeled images (e.g. DFUC2020) added to the "
             "3,994 unlabeled DFUC2021 images for pre-training.",
    )
    p.add_argument(
        "--img_size", type=int, default=224,
        help="Input resolution; must be a multiple of the patch size (14).",
    )
    p.add_argument("--lora_rank",   type=int,   default=16,
                   help="LoRA rank (alpha = 2 x rank).")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument(
        "--loss_fn", choices=["l1", "mse"], default="l1",
        help="Reconstruction loss function (default: l1).",
    )
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

# ── one epoch ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: DFUDinoLoRAForMIM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_samples  = 0

    for batch in loader:
        # Unlabeled DFUDataset returns a plain tensor (not a tuple).
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)

        loss, _pred, _mask = model(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n_samples  += images.size(0)

    return {"loss": total_loss / n_samples}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    debug      = args.debug
    epochs     = args.epochs     or (DEBUG_EPOCHS     if debug else DEFAULT_EPOCHS)
    batch_size = args.batch_size or (DEBUG_BATCH_SIZE  if debug else DEFAULT_BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"{'─' * 60}")
    print(f"  SimMIM Pre-training  (Domain-Adaptive Phase 1)")
    print(f"{'─' * 60}")
    print(f"Device     : {device}")
    print(f"Debug      : {debug}")
    print(f"Epochs     : {epochs}")
    print(f"Batch      : {batch_size}")
    print(f"Mask ratio : {args.mask_ratio}")
    print(f"Loss       : {args.loss_fn}")

    # ── data (unlabeled only) ────────────────────────────────────────────
    _, unlabeled_df = load_and_split_csv(CSV_PATH)
    print(f"Unlabeled imgs : {len(unlabeled_df)}")

    if debug:
        unlabeled_df = unlabeled_df.head(DEBUG_SAMPLES)
        print(f"Debug subset   : {len(unlabeled_df)} images")

    img_dir = IMG_DIR
    if args.img_size % 14:
        raise SystemExit("--img_size must be a multiple of 14")
    if args.extra_img_dirs:
        # Switch to absolute paths so images from several directories can be mixed.
        extra = []
        for d in args.extra_img_dirs:
            names = sorted(f for f in os.listdir(d)
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            print(f"Extra images   : {len(names):>6}  from {d}")
            extra += [os.path.abspath(os.path.join(d, f)) for f in names]
        base = unlabeled_df[["image"]].copy()
        base["image"] = [os.path.abspath(os.path.join(IMG_DIR, f)) for f in base["image"]]
        unlabeled_df = pd.concat([base, pd.DataFrame({"image": extra})], ignore_index=True)
        img_dir = ""
        print(f"Total pre-training images: {len(unlabeled_df)}")
        if debug:
            unlabeled_df = unlabeled_df.tail(DEBUG_SAMPLES)

    loader = build_loader(
        unlabeled_df, img_dir, get_train_transforms(args.img_size), batch_size,
        num_workers=args.num_workers,
    )

    # ── model ────────────────────────────────────────────────────────────
    model = DFUDinoLoRAForMIM(
        mask_ratio=args.mask_ratio,
        loss_fn=args.loss_fn,
        lora_rank=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
    )
    model.to(device)
    model.print_trainable_parameters()

    # ── optimiser ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    # ── training loop ────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(model, loader, optimizer, device)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss={metrics['loss']:.6f}  "
            f"({elapsed:.1f}s)"
        )

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Full pretrain model (backbone LoRA + decoder) — in case you want
    #    to resume pre-training later.
    full_path = os.path.join("checkpoints", "dfu_pretrain_mim.pt")
    torch.save(model.state_dict(), full_path)
    print(f"Full pretrain model  → {full_path}")

    # 2. Backbone LoRA weights only — load these into DFUDinoLoRA for
    #    Phase 2 fine-tuning:
    #        model = DFUDinoLoRA()
    #        model.backbone.load_state_dict(torch.load(backbone_path))
    backbone_path = os.path.join("checkpoints", "dfu_pretrained_backbone.pt")
    torch.save(model.encoder.backbone.state_dict(), backbone_path)
    print(f"Backbone LoRA weights → {backbone_path}")


if __name__ == "__main__":
    main()
