"""Metrics for the classification and rating-regression tasks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float | list]:
    """Accuracy, precision/recall/F1 for the positive class, ROC-AUC and the confusion matrix.

    The confusion matrix is ``[[TN, FP], [FN, TP]]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float | list] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    y_pred,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``metric(y_true, y_pred)`` over test rows."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def regression_metrics(y_true, y_pred) -> dict[str, float | None]:
    """MAE, RMSE, R^2, Spearman rho (None for constant predictions), and share within +-1 point."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rho = float(spearmanr(y_true, y_pred).statistic) if np.ptp(y_pred) > 0 else None
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": rho,
        "within_1": float(np.mean(np.abs(np.clip(np.rint(y_pred), 1, 10) - y_true) <= 1)),
    }
