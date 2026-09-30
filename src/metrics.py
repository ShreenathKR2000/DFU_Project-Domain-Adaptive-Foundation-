"""Classification metrics matching the DFUC2021 leaderboard columns."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.dataset import LABEL_COLS


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None
) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix and,
    when class probabilities are given, macro / micro AUC (``None`` if a class is
    absent from ``y_true``). Macro averages cover only classes present in ``y_true``."""
    ids = list(range(len(LABEL_COLS)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    present = support > 0
    macro_auc = micro_auc = None
    if probs is not None and present.all():
        onehot = np.eye(len(LABEL_COLS))[y_true]
        macro_auc = float(roc_auc_score(onehot, probs, average="macro"))
        micro_auc = float(roc_auc_score(onehot.ravel(), probs.ravel()))
    return {
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
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
