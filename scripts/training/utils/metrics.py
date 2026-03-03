"""Evaluation metrics for RAGE models."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_map(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Mean Average Precision for multi-label classification."""
    valid = labels.sum(axis=0) > 0
    if valid.sum() == 0:
        return 0.0
    return average_precision_score(
        labels[:, valid], predictions[:, valid], average="macro"
    )


def compute_roc_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Macro ROC-AUC for multi-label classification."""
    valid = labels.sum(axis=0) > 0
    if valid.sum() == 0:
        return 0.0
    return roc_auc_score(
        labels[:, valid], predictions[:, valid], average="macro"
    )


def concordance_correlation_coefficient(
    pred: np.ndarray, target: np.ndarray
) -> float:
    """Concordance Correlation Coefficient (CCC)."""
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var()
    target_var = target.var()
    covariance = np.mean((pred - pred_mean) * (target - target_mean))

    ccc = (2.0 * covariance) / (
        pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    )
    return float(ccc)
