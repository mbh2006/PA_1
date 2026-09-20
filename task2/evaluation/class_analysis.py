"""Per-class analysis helpers for target-domain evaluation (Tasks 2 and 3).

The assignment asks for class-level evidence: which classes improve or degrade
relative to the baseline and which confusions dominate.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from common.metrics import confusion_matrix, per_class_accuracy


def per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    accs = per_class_accuracy(y_true, y_pred, num_classes)
    rows: List[Dict[str, object]] = []
    for c in range(num_classes):
        total = int(cm[c, :].sum())
        rows.append(
            {
                "class": class_names[c],
                "support": total,
                "correct": int(cm[c, c]),
                "accuracy": float(accs[c]) if total else float("nan"),
            }
        )
    return rows


def top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
    k: int = 10,
) -> List[Dict[str, object]]:
    """Most frequent off-diagonal confusions, with the share of the true class."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    entries = []
    for t in range(num_classes):
        support = int(cm[t, :].sum())
        for p in range(num_classes):
            if t == p or cm[t, p] == 0:
                continue
            entries.append(
                {
                    "true_class": class_names[t],
                    "predicted_class": class_names[p],
                    "count": int(cm[t, p]),
                    "share_of_true_class": float(cm[t, p] / support) if support else 0.0,
                }
            )
    entries.sort(key=lambda row: row["count"], reverse=True)
    return entries[:k]


def delta_rows(
    baseline_rows: List[Dict[str, object]],
    adapted_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Per-class accuracy change (adapted - baseline), largest improvement first."""
    by_class = {row["class"]: row for row in baseline_rows}
    out = []
    for row in adapted_rows:
        base = by_class[row["class"]]["accuracy"]
        out.append(
            {
                "class": row["class"],
                "baseline_accuracy": base,
                "adapted_accuracy": row["accuracy"],
                "delta": row["accuracy"] - base,
            }
        )
    out.sort(key=lambda r: r["delta"], reverse=True)
    return out
