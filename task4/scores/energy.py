"""Energy score: u_Energy(x) = -log sum_k exp(z_k(x))."""
from __future__ import annotations

import numpy as np


def energy(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    maximum = z.max(axis=1, keepdims=True)
    return -(maximum[:, 0] + np.log(np.exp(z - maximum).sum(axis=1)))
