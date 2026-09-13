"""
Shared evaluation metrics (cefr_metrics.py) for the CEFR classification experiments.

Label convention (must match across experiments), the 8-level +/- scheme:
    A2=0, A2+=1, B1=2, B1+=3, B2=4, B2+=5, C1=6, C1+=7
QWK uses sklearn.cohen_kappa_score(weights="quadratic"); P/R/F1 default to macro
(every level weighted equally, which suits the imbalanced ordinal setting) with the
weighted variants also returned.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
    precision_score, recall_score,
)

LABEL_NAMES = ["A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+"]
LABELS = list(range(len(LABEL_NAMES)))      # [0..7]


def evaluate_predictions(y_true, y_pred, labels=LABELS) -> dict:
    """Full metric panel for integer predictions in {0..7}.

    `labels` is fixed so metrics stay stable even when a split's predictions miss a class.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    kw = dict(labels=labels, zero_division=0)
    return {
        "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic", labels=labels)),
        "mae": float(mean_absolute_error(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "adjacent_accuracy": float((np.abs(yt - yp) <= 1).mean()),   # off-by-one
        "precision_macro": float(precision_score(yt, yp, average="macro", **kw)),
        "recall_macro": float(recall_score(yt, yp, average="macro", **kw)),
        "f1_macro": float(f1_score(yt, yp, average="macro", **kw)),
        "precision_weighted": float(precision_score(yt, yp, average="weighted", **kw)),
        "recall_weighted": float(recall_score(yt, yp, average="weighted", **kw)),
        "f1_weighted": float(f1_score(yt, yp, average="weighted", **kw)),
    }
