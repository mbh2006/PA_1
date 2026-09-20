"""AUROC and validation-calibrated rejection metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """AUROC with the unknown class as the positive label."""
    scores = np.concatenate([np.asarray(known_scores), np.asarray(unknown_scores)])
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    return float(roc_auc_score(labels, scores))


def operating_point(u_known_val: np.ndarray, u_known_test: np.ndarray,
                    u_unknown: np.ndarray, percentile: float = 95.0) -> Dict[str, float]:
    """Threshold at the given percentile of unknownness on known validation data.

    Accept when u <= tau. Returns the achieved test-known acceptance rate, the
    unknown rejection rate and FPR@95TPR (= fraction of unknowns accepted).
    """
    tau = float(np.percentile(u_known_val, percentile))
    acceptance = float((np.asarray(u_known_test) <= tau).mean())
    rejection = float((np.asarray(u_unknown) > tau).mean())
    return {
        "threshold": tau,
        "known_test_acceptance": acceptance,
        "unknown_rejection": rejection,
        "fpr_at_95tpr": 1.0 - rejection,
    }
