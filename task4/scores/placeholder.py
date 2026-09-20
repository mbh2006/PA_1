"""PROSER placeholder-based unknownness score.

The open-set decision compares the strongest dummy response (plus the
validation-calibrated bias) with the strongest known-class response. A larger
score means the dummy wins, i.e. the input looks more unknown:

    u_placeholder(x) = (max_dummy z_d(x) + bias) - max_k z_k(x).

The bias is chosen on CIFAR-10 validation data so that the 95th percentile of
the unknownness equals zero, i.e. 95% of known validation examples are accepted
- the same convention as the assignment's threshold rule.
"""
from __future__ import annotations

import numpy as np


def calibration_bias(known_logits: np.ndarray, dummy_logits: np.ndarray,
                     percentile: float = 95.0) -> float:
    """Percentile of (dummy - known) on known validation data."""
    margin = np.asarray(dummy_logits).max(axis=1) - np.asarray(known_logits).max(axis=1)
    return float(np.percentile(margin, percentile))


def placeholder(known_logits: np.ndarray, dummy_logits: np.ndarray, bias: float) -> np.ndarray:
    best_known = np.asarray(known_logits).max(axis=1)
    best_dummy = np.asarray(dummy_logits).max(axis=1)
    return (best_dummy + bias) - best_known
