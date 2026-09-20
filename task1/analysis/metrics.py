"""Prediction-level metrics for Task 1 (scores, consistency, shape bias)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from common.metrics import evaluate_classification, mean_max_softmax


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


def classification_metrics(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[str, float]:
    predictions = np.asarray(logits).argmax(axis=1)
    metrics = evaluate_classification(labels, predictions, num_classes)
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "mean_max_confidence": mean_max_softmax(logits),
    }


def prediction_consistency(clean_logits: np.ndarray, transformed_logits: np.ndarray) -> float:
    clean = np.asarray(clean_logits).argmax(axis=1)
    transformed = np.asarray(transformed_logits).argmax(axis=1)
    return float((clean == transformed).mean())


def shape_texture_counts(predictions: np.ndarray, content_labels: np.ndarray,
                         style_labels: np.ndarray) -> Dict[str, int]:
    predictions = np.asarray(predictions)
    shape = int((predictions == np.asarray(content_labels)).sum())
    texture = int((predictions == np.asarray(style_labels)).sum())
    other = int(len(predictions) - shape - texture)
    return {"shape": shape, "texture": texture, "other": other,
            "total": int(len(predictions))}


def shape_bias_coverage(counts: Dict[str, int]) -> Dict[str, float]:
    decided = counts["shape"] + counts["texture"]
    total = counts["total"]
    return {
        "shape_bias": 100.0 * counts["shape"] / decided if decided else float("nan"),
        "coverage": 100.0 * decided / total if total else float("nan"),
    }
