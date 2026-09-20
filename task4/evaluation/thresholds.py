"""Rejection thresholds calibrated on known validation data only."""
from __future__ import annotations

import numpy as np


def threshold_at_percentile(unknownness_val: np.ndarray, percentile: float = 95.0) -> float:
    """tau = p-th percentile of the unknownness score on known validation data.

    Unknownness convention: larger = more novel, and an example is accepted
    when ``u(x) <= tau``.
    """
    return float(np.percentile(np.asarray(unknownness_val), percentile))
