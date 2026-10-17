"""
Re-balance ensemble probabilities so the predicted class mix matches the training labels.

Why: the first leaderboard submission (macro-F1 0.588, macro-AUC 0.870) showed that the
models rank images well but place the decision boundaries badly on the challenge images —
ischaemia was predicted for ~545 images while the test set holds only ~225 (precision 0.33),
and ~560 infections were missed. The test-set class mix is the same as training, so this
script finds one additive log-bias per class such that arg-max predictions on the unlabeled
test images reproduce the *training* label proportions, then renormalises.

No test labels are used: the targets come from train.csv counts only.

    python scripts/prior_match.py artifacts/sub_final_probs.csv --out sub_priormatch_probs.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

CLASSES = ["none", "infection", "ischaemia", "both"]
# DFUC2021 train.csv labeled counts (5,955 images).
TRAIN_COUNTS = {"none": 2552, "infection": 2555, "ischaemia": 227, "both": 621}


def fit_bias(logp: np.ndarray, target: np.ndarray, iters: int = 5000, lr: float = 0.05) -> np.ndarray:
    """Find per-class log-biases b so that the share of argmax(logp + b) per class ≈ target."""
    b = np.zeros(logp.shape[1])
    n = len(logp)
    for _ in range(iters):
        pred = np.bincount((logp + b).argmax(1), minlength=len(b)) / n
        b += lr * (np.log(target + 1e-6) - np.log(pred + 1e-6))
        b -= b.mean()
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probs_csv", help="*_probs.csv from src.predict (image,none,infection,ischaemia,both)")
    ap.add_argument("--out", default="sub_priormatch_probs.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.probs_csv)
    p = df[CLASSES].to_numpy(dtype=np.float64).clip(1e-8, 1)
    target = np.array([TRAIN_COUNTS[c] for c in CLASSES], dtype=float)
    target /= target.sum()

    b = fit_bias(np.log(p), target)
    q = p * np.exp(b)
    q /= q.sum(1, keepdims=True)

    before, after = p.argmax(1), q.argmax(1)
    n = len(df)
    print("log-bias per class:", dict(zip(CLASSES, b.round(3))))
    print(f"{'class':<10}{'target %':>10}{'before %':>10}{'after %':>10}")
    for i, c in enumerate(CLASSES):
        print(f"{c:<10}{100*target[i]:>10.1f}{100*(before==i).mean():>10.1f}{100*(after==i).mean():>10.1f}")
    print(f"changed predictions: {(before != after).sum()} of {n}")
    print("moves (from -> to):")
    moves = pd.crosstab(pd.Series(before).map(dict(enumerate(CLASSES))),
                        pd.Series(after).map(dict(enumerate(CLASSES))))
    print(moves)

    out = pd.DataFrame(q, columns=CLASSES)
    out.insert(0, "image", df["image"])
    out.to_csv(a.out, index=False, float_format="%.8f")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
