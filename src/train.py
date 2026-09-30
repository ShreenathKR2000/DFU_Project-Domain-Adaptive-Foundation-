"""
Training loop for DFU classification.

The labeled data is split (stratified) into train / val / test.  The best epoch
is chosen on validation macro-F1 and the final numbers are reported on the
held-out test split, so runs with and without Phase-1 LoRA weights can be
compared fairly.  Runs that share ``--seed`` share the same split.

Usage
-----
Baseline (no domain adaptation):
    python -m src.train --seed 0 --run_name scratch_seed0

With Phase-1 LoRA weights:
    python -m src.train --seed 0 --run_name pretrained_seed0 \
        --pretrained_lora_path checkpoints/dfu_pretrained_backbone.pt

Debug run (tiny subset, batch_size=2, 2 epochs — fits GTX 1650 4 GB):
    python -m src.train --debug
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import (
    DFUDataset,
    LABEL_COLS,
    check_labeled,
    class_counts,
    get_eval_transforms,
    get_train_transforms,
    labels_from_df,
    load_and_split_csv,
    load_groups,
    make_splits,
    subsample_stratified,
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
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument(
        "--split_seed", type=int, default=None,
        help="Seed for the train/val/test split (default: same as --seed).",
    )
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument(
        "--train_frac", type=float, default=1.0,
        help="Fraction of the training split to keep (class-stratified, same "
             "subset for every run sharing --split_seed). Val/test are unchanged.",
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Input resolution (multiple of 14). DINOv2 interpolates its position "
             "embeddings; cost grows roughly with (size/224)^2.",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank (alpha = 2 x rank). Must match the Phase-1 checkpoint.",
    )
    parser.add_argument(
        "--group_csv", type=str, default=None,
        help="image,group CSV from `python -m src.make_groups`; keeps near-duplicate "
             "images in the same split (leak-free validation/test).",
    )
    parser.add_argument(
        "--aug", choices=["basic", "strong"], default="basic",
        help="Training augmentation strength.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--final", action="store_true",
        help="Leaderboard mode: train on ALL labeled images (no val/test holdout) for "
             "a fixed --epochs and save the last-epoch weights to "
             "checkpoints/<run_name>_final.pt. Pick --epochs from held-out runs first.",
    )
    parser.add_argument(
        "--imbalance", choices=["none", "weights", "sampler"], default="none",
        help="Class-imbalance handling: inverse-frequency CE weights, or a "
             "class-balanced sampler.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Name used for the results JSON and best checkpoint.",
    )
    parser.add_argument("--out_dir", type=str, default="results")
    return parser.parse_args()


def set_seed(seed: int) -> None:
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


def class_weights(train_df) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 over the classes."""
    counts = np.bincount(labels_from_df(train_df), minlength=len(LABEL_COLS))
    w = counts.sum() / (len(LABEL_COLS) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


# ── one epoch ────────────────────────────────────────────────────────────────

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


# ── evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1 and confusion matrix.

    Macro averages only cover classes present in the evaluated split.
    """
    model.eval()
    preds, targets = [], []
    for images, labels in loader:
        logits, _ = model(images.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(labels)
    return compute_metrics(torch.cat(targets).numpy(), torch.cat(preds).numpy())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1 and confusion matrix."""
    ids = list(range(len(LABEL_COLS)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    present = support > 0
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(rec[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": {
            name: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(LABEL_COLS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def recall_str(metrics: dict) -> str:
    return " ".join(
        f"{name[:4]}={m['recall']:.2f}" for name, m in metrics["per_class"].items()
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    debug = args.debug
    epochs = args.epochs or (DEBUG_EPOCHS if debug else DEFAULT_EPOCHS)
    batch_size = args.batch_size or (DEBUG_BATCH_SIZE if debug else DEFAULT_BATCH_SIZE)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    arm = "pretrained" if args.pretrained_lora_path else "scratch"
    run_name = args.run_name or f"{arm}_seed{args.seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"Device    : {device}")
    print(f"Debug     : {debug}")
    print(f"Epochs    : {epochs}")
    print(f"Batch     : {batch_size}")
    print(f"LoRA      : {args.pretrained_lora_path or 'scratch'}")
    print(f"Seed      : {args.seed} (split seed {split_seed})")
    print(f"Imbalance : {args.imbalance}")
    print(f"Train frac: {args.train_frac}")
    print(f"Img size  : {args.img_size}   LoRA rank: {args.lora_rank}")
    print(f"Aug       : {args.aug}   label smoothing: {args.label_smoothing}")
    print(f"Run name  : {run_name}")

    # ── data ─────────────────────────────────────────────────────────────
    labeled_df, _ = load_and_split_csv(CSV_PATH)
    check_labeled(labeled_df)

    if debug:
        labeled_df = labeled_df.head(DEBUG_SAMPLES)

    if args.final:
        # Leaderboard mode: every labeled image is used for training.
        train_df, val_df, test_df = labeled_df, None, None
        print("FINAL mode: training on all labeled images, no holdout evaluation")
    else:
        groups = load_groups(labeled_df, args.group_csv) if args.group_csv else None
        if groups is not None:
            print(f"Group-aware split using {args.group_csv} "
                  f"({len(set(groups.tolist()))} groups)")
        train_df, val_df, test_df = make_splits(
            labeled_df, args.val_frac, args.test_frac, seed=split_seed, groups=groups
        )
        if args.train_frac < 1.0:
            train_df = subsample_stratified(train_df, args.train_frac, seed=split_seed)
            print(f"Low-label setting: keeping {args.train_frac:.0%} of the training split")
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        if part is not None:
            print(f"{name:<5} split: {len(part):>5}  {class_counts(part)}")

    sampler = None
    if args.imbalance == "sampler":
        y_train = labels_from_df(train_df)
        per_class = class_weights(train_df).numpy()
        sampler = WeightedRandomSampler(
            weights=per_class[y_train].tolist(),
            num_samples=len(y_train),
            replacement=True,
        )

    train_loader = build_loader(
        train_df, IMG_DIR, get_train_transforms(args.img_size, strong=args.aug == "strong"), batch_size,
        sampler=sampler, num_workers=args.num_workers,
    )
    eval_bs = max(batch_size, 32)
    if not args.final:
        val_loader = build_loader(
            val_df, IMG_DIR, get_eval_transforms(args.img_size), eval_bs,
            shuffle=False, num_workers=args.num_workers,
        )
        test_loader = build_loader(
            test_df, IMG_DIR, get_eval_transforms(args.img_size), eval_bs,
            shuffle=False, num_workers=args.num_workers,
        )

    # ── model ────────────────────────────────────────────────────────────
    model = DFUDinoLoRA(
        pretrained_lora_path=args.pretrained_lora_path,
        lora_rank=args.lora_rank, lora_alpha=2 * args.lora_rank,
    )
    model.to(device)
    model.print_trainable_parameters()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    # ── optimiser & loss ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    ce_weight = class_weights(train_df).to(device) if args.imbalance == "weights" else None
    ce_criterion = nn.CrossEntropyLoss(
        weight=ce_weight, label_smoothing=args.label_smoothing
    )
    con_criterion = SupConLoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join("checkpoints", f"{run_name}_best.pt")

    # ── training loop ────────────────────────────────────────────────────
    history = []
    best_f1, best_epoch = -1.0, 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, con_criterion, device
        )
        if args.final:
            print(
                f"Epoch {epoch:>3}/{epochs}  loss={metrics['loss']:.4f}  "
                f"ce={metrics['ce_loss']:.4f}  con={metrics['con_loss']:.4f}  "
                f"train_acc={metrics['accuracy']:.3f}  ({time.time() - t0:.1f}s)"
            )
            history.append({"epoch": epoch, "train": metrics})
            continue
        val = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"ce={metrics['ce_loss']:.4f}  "
            f"con={metrics['con_loss']:.4f}  "
            f"train_acc={metrics['accuracy']:.3f}  "
            f"val_acc={val['accuracy']:.3f}  "
            f"val_f1={val['macro_f1']:.3f}  "
            f"recall[{recall_str(val)}]  "
            f"({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch, "train": metrics, "val": val})

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable},
                best_path,
            )

    if args.final:
        final_path = os.path.join("checkpoints", f"{run_name}_final.pt")
        torch.save(
            {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable},
            final_path,
        )
        result = {
            "run_name": run_name, "arm": arm, "mode": "final",
            "config": {**vars(args), "epochs": epochs, "batch_size": batch_size},
            "train_size": len(train_df), "train_class_counts": class_counts(train_df),
            "history": history, "train_seconds": time.time() - t_start,
        }
        with open(os.path.join(args.out_dir, f"{run_name}_final.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFinal weights → {final_path}")
        return

    # ── final evaluation on the held-out test split ──────────────────────
    # Last-epoch model: needs no validation labels for model selection, which
    # matters in the low-label setting (the val split stays full-size).
    test_last = evaluate(model, test_loader, device)
    model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
    test = evaluate(model, test_loader, device)
    print(f"\nBest epoch {best_epoch} (val macro-F1 {best_f1:.4f}) → test:")
    print(
        f"  acc={test['accuracy']:.3f}  bal_acc={test['balanced_accuracy']:.3f}  "
        f"macro_f1={test['macro_f1']:.4f}"
    )
    print(f"  last epoch ({epochs}): macro_f1={test_last['macro_f1']:.4f}  "
          f"acc={test_last['accuracy']:.3f}")
    for name, m in test["per_class"].items():
        print(
            f"  {name:<10} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  n={m['support']}"
        )
    print(f"  confusion matrix (rows=true, cols=pred {LABEL_COLS}):")
    for row in test["confusion_matrix"]:
        print("   ", row)

    result = {
        "run_name": run_name,
        "arm": arm,
        "config": {**vars(args), "epochs": epochs, "batch_size": batch_size,
                   "split_seed": split_seed},
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "split_class_counts": {
            "train": class_counts(train_df),
            "val": class_counts(val_df),
            "test": class_counts(test_df),
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test": test,
        "test_last_epoch": test_last,
        "history": history,
        "train_seconds": time.time() - t_start,
    }
    result_path = os.path.join(args.out_dir, f"{run_name}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBest weights → {best_path}")
    print(f"Results      → {result_path}")


if __name__ == "__main__":
    main()
