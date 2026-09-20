"""Inspect the unknown images that the frozen model wrongly accepts."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def accepted_unknowns(
    unknownness: np.ndarray,
    threshold: float,
    fine_labels: np.ndarray,
    fine_names: Sequence[str],
    known_logits: np.ndarray,
    known_names: Sequence[str],
    top_k: int = 8,
) -> List[Dict[str, object]]:
    """Rows for the unknown examples with the lowest unknownness (accepted most confidently)."""
    unknownness = np.asarray(unknownness)
    accepted = np.where(unknownness <= threshold)[0]
    order = accepted[np.argsort(unknownness[accepted])]
    rows: List[Dict[str, object]] = []
    for index in order[:top_k]:
        rows.append(
            {
                "unknown_class": fine_names[int(fine_labels[index])],
                "predicted_known_class": known_names[int(np.argmax(known_logits[index]))],
                "score": float(unknownness[index]),
                "threshold": float(threshold),
            }
        )
    return rows
