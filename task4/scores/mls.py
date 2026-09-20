"""Maximum logit score: u_MLS(x) = -max_k z_k(x).

Keeps the absolute logit magnitude that the softmax discards.
"""
from __future__ import annotations

import numpy as np


def mls(logits: np.ndarray) -> np.ndarray:
    return -np.asarray(logits, dtype=np.float64).max(axis=1)
