"""Classification metrics implemented with numpy only (no hidden dependencies).

Task 2/3 select checkpoints using *mean macro-F1 across the source validation
domains*; Task 4 reports accuracy and needs per-class accuracy / confusions.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = ground truth, columns = prediction."""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Unweighted mean of per-class F1 scores."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(f1s))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    out = np.full(num_classes, np.nan)
    for c in range(num_classes):
        total = cm[c, :].sum()
        if total > 0:
            out[c] = cm[c, c] / total
    return out


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, object]:
    """One-stop metric bundle used by tasks 2-4."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "per_class_accuracy": per_class_accuracy(y_true, y_pred, num_classes).tolist(),
        "confusion_matrix": cm.tolist(),
        "num_samples": int(np.asarray(y_true).size),
    }


def mean_max_softmax(logits: np.ndarray) -> float:
    """Mean maximum softmax probability; used by Task 1 confidence reporting."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return float(p.max(axis=1).mean())


def mean_confidence_from_scores(scores: np.ndarray) -> float:
    """Same idea when the model gives similarity scores instead of logits."""
    return mean_max_softmax(scores)
