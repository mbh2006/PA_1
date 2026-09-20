"""PROSER placeholder-based unknownness score.

The open-set decision compares the strongest known-class response with the
strongest dummy response (plus the validation-calibrated bias):

    u_placeholder(x) = max_k z_k(x) - (max_dummy z_d(x) + bias).

The bias is chosen on CIFAR-10 validation data so that the 95th percentile of
the margin equals zero, i.e. 95% of known validation examples are accepted -
the same convention as the assignment's threshold rule.
"""
from __future__ import annotations

import numpy as np


def calibration_bias(known_logits: np.ndarray, dummy_logits: np.ndarray,
                     percentile: float = 95.0) -> float:
    margin = np.asarray(known_logits).max(axis=1) - np.asarray(dummy_logits).max(axis=1)
    return float(np.percentile(margin, percentile))


def placeholder(known_logits: np.ndarray, dummy_logits: np.ndarray, bias: float) -> np.ndarray:
    best_known = np.asarray(known_logits).max(axis=1)
    best_dummy = np.asarray(dummy_logits).max(axis=1)
    return best_known - (best_dummy + bias)
