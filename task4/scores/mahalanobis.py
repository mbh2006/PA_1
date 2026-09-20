"""Mahalanobis score: distance from the nearest known-class feature cluster.

Class means mu_c and one shared *diagonal* covariance Sigma are estimated from
unaugmented CIFAR-10 training features, with 1e-6 added to every diagonal entry.
The score is

    u_Mah(x) = min_c (f(x) - mu_c)^T Sigma^{-1} (f(x) - mu_c).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_mahalanobis(features: np.ndarray, labels: np.ndarray, num_classes: int,
                    epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    means = np.zeros((num_classes, features.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        means[c] = features[labels == c].mean(axis=0)

    covariance = np.zeros(features.shape[1], dtype=np.float64)
    for c in range(num_classes):
        residuals = features[labels == c] - means[c]
        covariance += (residuals ** 2).mean(axis=0)
    covariance /= num_classes
    covariance += epsilon
    return means, covariance


def mahalanobis(features: np.ndarray, means: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    inverse = 1.0 / covariance
    distances = np.stack(
        [((features - mean) ** 2 * inverse).sum(axis=1) for mean in means], axis=1
    )
    return distances.min(axis=1)
