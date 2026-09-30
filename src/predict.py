"""
Inference with test-time augmentation (TTA) and checkpoint ensembling.

Two modes
---------
1. Evaluate on a held-out split of the labeled data, to measure what TTA and
   ensembling buy you (same split protocol as ``src.train``):

       python -m src.predict --checkpoints checkpoints/a_best.pt checkpoints/b_best.pt \
           --split test --split_seed 0 --tta

   IMPORTANT: every checkpoint must have been trained with the same
   ``--split_seed`` (vary ``--seed`` instead), otherwise some checkpoints have
   seen the test images and the ensemble score is inflated.

2. Predict unlabeled images (e.g. a challenge test set) and write CSVs:

       python -m src.predict --checkpoints checkpoints/*_final.pt \
           --img_dir /path/to/test_images --tta --out predictions.csv

   ``predictions.csv`` has ``image,none,infection,ischaemia,both`` with a one-hot
   arg-max prediction (same schema as train.csv); ``predictions_probs.csv`` holds
   the averaged class probabilities. Adapt the column layout to the challenge's
   submission format if it differs.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import (
    DFUDataset,
    LABEL_COLS,
    check_labeled,
    get_eval_transforms,
    labels_from_df,
    load_and_split_csv,
    make_splits,
)
from src.model import DFUDinoLoRA
from src.train import CSV_PATH, IMG_DIR, compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTA / ensemble inference")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="*_best.pt / *_final.pt files written by src.train")
    p.add_argument("--tta", action="store_true",
                   help="Average over identity + horizontal / vertical / both flips.")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lora_rank", type=int, default=16,
                   help="Must match the rank the checkpoints were trained with.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", choices=["val", "test"],
                     help="Evaluate on a split of the labeled DFUC2021 data.")
    src.add_argument("--img_dir", type=str,
                     help="Directory of unlabeled images to predict.")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--out", type=str, default="predictions.csv")
    return p.parse_args()


def load_model(path: str, device: torch.device, lora_rank: int = 16) -> DFUDinoLoRA:
    """Rebuild the model and load the trainable (LoRA + head) weights."""
    model = DFUDinoLoRA(lora_rank=lora_rank, lora_alpha=2 * lora_rank)
    state = torch.load(path, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"{path}: unexpected keys {result.unexpected_keys[:3]}...")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = trainable - set(state)
    if missing:
        raise RuntimeError(f"{path}: missing trainable weights, e.g. {sorted(missing)[:3]}")
    return model.to(device).eval()


@torch.no_grad()
def predict_probs(
    model: DFUDinoLoRA, loader: DataLoader, device: torch.device, tta: bool
) -> np.ndarray:
    """Softmax probabilities, averaged over flips when ``tta`` is set."""
    out = []
    for batch in loader:
        images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        views = [images]
        if tta:
            views += [images.flip(3), images.flip(2), images.flip(2).flip(3)]
        probs = sum(F.softmax(model(v)[0].float(), dim=1) for v in views) / len(views)
        out.append(probs.cpu())
    return torch.cat(out).numpy()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   checkpoints: {len(args.checkpoints)}   TTA: {args.tta}")

    if args.split:
        labeled_df, _ = load_and_split_csv(CSV_PATH)
        check_labeled(labeled_df)
        _, val_df, test_df = make_splits(
            labeled_df, args.val_frac, args.test_frac, seed=args.split_seed
        )
        df, img_dir = (val_df if args.split == "val" else test_df), IMG_DIR
        y_true = labels_from_df(df)
        print(f"Evaluating on the '{args.split}' split ({len(df)} images, "
              f"split seed {args.split_seed})")
    else:
        names = sorted(f for f in os.listdir(args.img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        df, img_dir, y_true = pd.DataFrame({"image": names}), args.img_dir, None
        print(f"Predicting {len(df)} images from {args.img_dir}")

    loader = DataLoader(
        DFUDataset(df, img_dir, transform=get_eval_transforms(args.img_size)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    all_probs: list[np.ndarray] = []
    for path in args.checkpoints:
        model = load_model(path, device, args.lora_rank)
        probs = predict_probs(model, loader, device, args.tta)
        all_probs.append(probs)
        if y_true is not None:
            m = compute_metrics(y_true, probs.argmax(1))
            print(f"  {os.path.basename(path):<40} macro-F1={m['macro_f1']:.4f}  "
                  f"acc={m['accuracy']:.3f}")
        del model
        torch.cuda.empty_cache()

    ensemble = np.mean(all_probs, axis=0)

    if y_true is not None:
        m = compute_metrics(y_true, ensemble.argmax(1))
        label = "ensemble" if len(all_probs) > 1 else "single model"
        print(f"\n{label}{' + TTA' if args.tta else ''}: "
              f"macro-F1={m['macro_f1']:.4f}  bal_acc={m['balanced_accuracy']:.3f}  "
              f"acc={m['accuracy']:.3f}")
        for name, c in m["per_class"].items():
            print(f"  {name:<10} P={c['precision']:.3f}  R={c['recall']:.3f}  "
                  f"F1={c['f1']:.3f}  n={c['support']}")
        return

    pred = ensemble.argmax(1)
    onehot = pd.DataFrame(np.eye(len(LABEL_COLS), dtype=int)[pred], columns=LABEL_COLS)
    out = pd.concat([df[["image"]].reset_index(drop=True), onehot], axis=1)
    out.to_csv(args.out, index=False)
    probs_path = os.path.splitext(args.out)[0] + "_probs.csv"
    pd.concat([df[["image"]].reset_index(drop=True),
               pd.DataFrame(ensemble, columns=LABEL_COLS)], axis=1).to_csv(probs_path, index=False)
    print(f"Predictions   → {args.out}\nProbabilities → {probs_path}")
    print("Predicted class counts:", dict(zip(LABEL_COLS, np.bincount(pred, minlength=4).tolist())))


if __name__ == "__main__":
    main()
