"""Maximum softmax probability score: u_MSP(x) = 1 - max_k p_k(x)."""
from __future__ import annotations

import numpy as np


def msp(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    probabilities = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return 1.0 - probabilities.max(axis=1)
