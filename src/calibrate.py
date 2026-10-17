"""
Decision-rule calibration, validated on the leak-free split *before* touching the leaderboard.

The models rank images well (high AUC) but the arg-max decision can sit in the wrong place.
This module compares three ways of turning the ensemble probabilities into labels, using only
data we have labels for, and applies the winner to the challenge predictions according to a
rule fixed in advance (``--rule``, see ``RULES``).

Methods
-------
none      plain arg-max (what submission 1 used).
prior     label-free prior matching: one additive log-bias per class so that the arg-max
          predictions on the *set being predicted* reproduce the training label proportions.
          Uses no labels of that set — only its predictions — so it can be applied to the
          challenge images exactly as it is evaluated here.
val_bias  per-class log-biases tuned to maximise macro-F1 on the *validation* split (labels),
          then frozen and applied unchanged to the test split / challenge images.

Protocol
--------
1. ``python -m src.predict ... --split val  --tta --out results/calib/val.csv``
2. ``python -m src.predict ... --split test --tta --out results/calib/test.csv``
3. ``python -m src.calibrate --val results/calib/val.csv --test results/calib/test.csv \\
        --challenge artifacts/sub_final_probs.csv --rule noninferiority --out_dir results/calib``

The internal test split is used once, for the comparison; the decision rule then picks the
method without any leaderboard information. ``scripts/run_calibration.sh`` runs all three steps.
Runs on CPU in seconds (no torch needed).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLASSES = ["none", "infection", "ischaemia", "both"]
# DFUC2021 train.csv labeled counts (5,955 images); the stratified splits share this mix.
TRAIN_COUNTS = np.array([2552, 2555, 227, 621], dtype=float)

RULES = {
    "superiority": (
        "Adopt the method with the largest mean macro-F1 gain over 'none' on the internal test "
        "split, but only if that gain is > 0 in at least 90% of paired bootstrap resamples; "
        "otherwise keep 'none' (plain arg-max)."
    ),
    "noninferiority": (
        "Adopt 'prior' if, on the internal test split, it is non-inferior to 'none': mean gain "
        ">= 0 and the one-sided 95% lower bound of the gain (5th bootstrap percentile) > -0.02 "
        "macro-F1. Rationale: prior "
        "matching targets a shift between training and challenge images, which the internal "
        "split does not contain, so internally it can only be shown not to hurt. 'val_bias' "
        "is adopted instead only if it beats 'none' in >= 90% of resamples AND beats 'prior' on "
        "mean gain. Otherwise keep 'none'."
    ),
}
METHODS = ["none", "prior", "val_bias"]


# ── metrics (fast, numpy only, so the bootstrap is cheap) ────────────────────

def f1_per_class(y: np.ndarray, pred: np.ndarray, k: int = 4) -> np.ndarray:
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)  # predicted + true counts
    return np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_per_class(y, pred).mean())


def uar(y: np.ndarray, pred: np.ndarray, k: int = 4) -> float:
    return float(np.mean([(pred[y == c] == c).mean() for c in range(k) if (y == c).any()]))


# ── decision rules ───────────────────────────────────────────────────────────

def apply_bias(p: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Probabilities after adding log-biases ``b`` and renormalising."""
    q = np.log(np.clip(p, 1e-8, 1)) + b
    q = np.exp(q - q.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def fit_prior_bias(p: np.ndarray, target: np.ndarray, iters: int = 5000, lr: float = 0.05) -> np.ndarray:
    """Log-biases such that the arg-max class shares of ``p`` match ``target`` (no labels)."""
    logp = np.log(np.clip(p, 1e-8, 1))
    target = target / target.sum()
    b = np.zeros(p.shape[1])
    for _ in range(iters):
        share = np.bincount((logp + b).argmax(1), minlength=len(b)) / len(p)
        b += lr * (np.log(target + 1e-6) - np.log(share + 1e-6))
        b -= b.mean()
    return b


def fit_val_bias(p: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None, passes: int = 3) -> np.ndarray:
    """Coordinate ascent on per-class log-biases to maximise macro-F1 on labeled data.
    Class 0 is the reference (bias 0). A coarse grid limits over-fitting to the small
    rare-class counts of the validation split."""
    grid = np.arange(-3.0, 3.01, 0.1) if grid is None else grid
    logp = np.log(np.clip(p, 1e-8, 1))
    b = np.zeros(p.shape[1])
    best = macro_f1(y, logp.argmax(1))
    for _ in range(passes):
        improved = False
        for c in range(1, p.shape[1]):
            for v in grid:
                trial = b.copy()
                trial[c] = v
                f = macro_f1(y, (logp + trial).argmax(1))
                if f > best + 1e-9:
                    best, b, improved = f, trial, True
        if not improved:
            break
    return b


# ── evaluation ───────────────────────────────────────────────────────────────

def summarise(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    onehot = np.eye(len(CLASSES))[y]
    return {
        "macro_f1": macro_f1(y, pred),
        "uar": uar(y, pred),
        "macro_auc": float(roc_auc_score(onehot, p, average="macro")),
        "f1": dict(zip(CLASSES, f1_per_class(y, pred).round(4).tolist())),
        "predicted": dict(zip(CLASSES, np.bincount(pred, minlength=4).tolist())),
        "true": dict(zip(CLASSES, np.bincount(y, minlength=4).tolist())),
    }


def paired_bootstrap(y, pred_a, pred_b, n: int = 2000, seed: int = 0) -> dict:
    """Macro-F1(b) − macro-F1(a) over resamples of the test images."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, len(y), len(y))
        diffs[i] = macro_f1(y[ix], pred_b[ix]) - macro_f1(y[ix], pred_a[ix])
    return {
        "mean_gain": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p05": float(np.percentile(diffs, 5)),
        "p_gain_positive": float((diffs > 0).mean()),
    }


def read_split(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "label" not in df:
        raise ValueError(f"{path}: no 'label' column — create it with src.predict --split ... --out")
    return df[CLASSES].to_numpy(float), df["label"].to_numpy(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val", required=True, help="val-split probabilities + labels (src.predict --split val --out)")
    ap.add_argument("--test", required=True, help="test-split probabilities + labels (src.predict --split test --out)")
    ap.add_argument("--challenge", help="challenge *_probs.csv to calibrate with the chosen method")
    ap.add_argument("--out_dir", default="results/calib")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--rule", choices=sorted(RULES), required=True,
                    help="Decision rule — choose and record it BEFORE running (see RULES).")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    p_val, y_val = read_split(a.val)
    p_test, y_test = read_split(a.test)

    # Fit every method without test labels.
    b_val = fit_val_bias(p_val, y_val)
    b_prior_test = fit_prior_bias(p_test, TRAIN_COUNTS)
    calibrated = {
        "none": p_test,
        "prior": apply_bias(p_test, b_prior_test),
        "val_bias": apply_bias(p_test, b_val),
    }

    print(f"Internal test split: {len(y_test)} images; validation: {len(y_val)} images\n")
    print(f"{'method':<10}{'macro-F1':>9}{'UAR':>7}{'AUC':>7}   F1 none/inf/isch/both     predicted counts")
    report = {"rule": a.rule, "decision_rule": RULES[a.rule], "biases": {
        "val_bias": dict(zip(CLASSES, b_val.round(3).tolist())),
        "prior_on_test": dict(zip(CLASSES, b_prior_test.round(3).tolist())),
    }, "methods": {}}
    base_pred = p_test.argmax(1)
    for m in METHODS:
        s = summarise(y_test, calibrated[m])
        if m != "none":
            s["vs_none"] = paired_bootstrap(y_test, base_pred, calibrated[m].argmax(1), a.bootstrap)
        report["methods"][m] = s
        f1s = " / ".join(f"{v:.3f}" for v in s["f1"].values())
        cnt = " / ".join(str(v) for v in s["predicted"].values())
        print(f"{m:<10}{s['macro_f1']:>9.4f}{s['uar']:>7.3f}{s['macro_auc']:>7.3f}   {f1s}   {cnt}")
    print("true counts:", " / ".join(str(v) for v in report["methods"]["none"]["true"].values()))

    print("\nPaired bootstrap vs 'none' (macro-F1 gain on the internal test split):")
    for m in METHODS[1:]:
        bs = report["methods"][m]["vs_none"]
        print(f"  {m:<9} mean {bs['mean_gain']:+.4f}   95% CI [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}]"
              f"   gain > 0 in {100 * bs['p_gain_positive']:.0f}% of resamples")

    # Pre-registered decision.
    bs = {m: report["methods"][m]["vs_none"] for m in METHODS[1:]}
    vb_ok = bs["val_bias"]["mean_gain"] > 0 and bs["val_bias"]["p_gain_positive"] >= 0.90
    if a.rule == "superiority":
        eligible = [m for m in METHODS[1:]
                    if bs[m]["mean_gain"] > 0 and bs[m]["p_gain_positive"] >= 0.90]
        chosen = max(eligible, key=lambda m: bs[m]["mean_gain"]) if eligible else "none"
    else:
        prior_ok = bs["prior"]["mean_gain"] >= 0 and bs["prior"]["p05"] > -0.02
        if vb_ok and bs["val_bias"]["mean_gain"] > bs["prior"]["mean_gain"]:
            chosen = "val_bias"
        elif prior_ok:
            chosen = "prior"
        else:
            chosen = "val_bias" if vb_ok else "none"
    report["chosen"] = chosen
    print(f"\nDecision rule ({a.rule}): {RULES[a.rule]}\n→ chosen method: {chosen}")

    if a.challenge:
        df = pd.read_csv(a.challenge)
        p = df[CLASSES].to_numpy(float)
        if chosen == "prior":
            b = fit_prior_bias(p, TRAIN_COUNTS)
        elif chosen == "val_bias":
            b = b_val
        else:
            b = np.zeros(len(CLASSES))
        q = apply_bias(p, b)
        out = pd.DataFrame(q, columns=CLASSES)
        out.insert(0, "image", df["image"])
        path = os.path.join(a.out_dir, f"challenge_{chosen}_probs.csv")
        out.to_csv(path, index=False, float_format="%.8f")
        before = np.bincount(p.argmax(1), minlength=4)
        after = np.bincount(q.argmax(1), minlength=4)
        report["challenge"] = {
            "input": a.challenge, "output": path,
            "bias": dict(zip(CLASSES, b.round(3).tolist())),
            "predicted_before": dict(zip(CLASSES, before.tolist())),
            "predicted_after": dict(zip(CLASSES, after.tolist())),
            "changed": int((p.argmax(1) != q.argmax(1)).sum()),
        }
        print(f"\nChallenge predictions ({len(df)} images) with '{chosen}':")
        print("  predicted before:", dict(zip(CLASSES, before.tolist())))
        print("  predicted after: ", dict(zip(CLASSES, after.tolist())))
        print(f"  changed: {report['challenge']['changed']}   → {path}")
        if chosen == "none":
            print("  (identical to the input: no method met the decision rule — do not resubmit)")

    with open(os.path.join(a.out_dir, "calibration.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport → {os.path.join(a.out_dir, 'calibration.json')}")


if __name__ == "__main__":
    main()
