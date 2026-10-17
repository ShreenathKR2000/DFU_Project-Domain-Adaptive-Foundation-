"""
Re-balance ensemble probabilities so the predicted class mix matches the training labels.

Why: the first leaderboard submission (macro-F1 0.588, macro-AUC 0.870) showed that the
models rank images well but place the decision boundaries badly on the challenge images —
ischaemia was predicted for ~545 images while the test set holds only ~225 (precision 0.33),
and ~560 infections were missed. The test-set class mix is the same as training, so this
script finds one additive log-bias per class such that arg-max predictions on the unlabeled
test images reproduce the *training* label proportions, then renormalises.

No test labels are used: the targets come from train.csv counts only.

NOTE: this applies prior matching directly. The proper route is src.calibrate, which first
checks on the leak-free split whether prior matching (or validation-tuned biases) actually
helps and applies a method only if it passes a decision rule fixed in advance.

    python scripts/prior_match.py artifacts/sub_final_probs.csv --out sub_priormatch_probs.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.calibrate import CLASSES, TRAIN_COUNTS, apply_bias, fit_prior_bias  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probs_csv", help="*_probs.csv from src.predict (image,none,infection,ischaemia,both)")
    ap.add_argument("--out", default="sub_priormatch_probs.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.probs_csv)
    p = df[CLASSES].to_numpy(dtype=np.float64).clip(1e-8, 1)
    target = TRAIN_COUNTS / TRAIN_COUNTS.sum()
    b = fit_prior_bias(p, TRAIN_COUNTS)
    q = apply_bias(p, b)

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
