"""Classification metrics used across the research scripts.

Parkinson's (label 1) is the positive class, so recall = sensitivity
(share of Parkinson's recordings detected) and specificity is the share of
healthy recordings correctly identified.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

SCALAR_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
)


def positive_scores(model: Pipeline, X: ArrayLike) -> np.ndarray:
    """Return a continuous score for the positive class, for ROC-AUC.

    Uses `decision_function` when available (e.g. SVC without probability
    calibration), otherwise the predicted probability of class 1.
    """
    # Pipeline only exposes decision_function if its final step has one.
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return np.asarray(model.predict_proba(X))[:, 1]


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike, y_score: ArrayLike) -> dict[str, float]:
    """Compute every metric reported in the evaluation tables.

    Returns the scalar metrics in `SCALAR_METRICS` plus the confusion-matrix
    counts `tn`, `fp`, `fn`, `tp`. Precision is 0 when nothing is predicted
    positive. ROC-AUC is NaN when `y_true` has a single class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(y_true)) < 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_fitted(model: Pipeline, X: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Score an already-fitted model on (X, y)."""
    return compute_metrics(y, model.predict(X), positive_scores(model, X))
