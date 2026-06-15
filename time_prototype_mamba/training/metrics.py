from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def binary_classification_metrics(
    labels: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.float32).reshape(-1)
    y_prob = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        return {
            "auc": math.nan,
            "acc": math.nan,
            "f1": math.nan,
            "sensitivity": math.nan,
            "specificity": math.nan,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = math.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": auc,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

