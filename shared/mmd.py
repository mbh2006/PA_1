"""Maximum Mean Discrepancy with a sum of RBF kernels (DAN / DAN-DG).

PA-1 specification:
    L_DAN = L_cls + lambda * || E_s[phi(F(x_s))] - E_t[phi(F(x_t))] ||_H^2
with three RBF kernels whose bandwidths are 0.5, 1 and 2 times the median
pairwise *squared* feature distance of the current combined batch.

The kernel trick means phi is never built; the squared MMD is estimated from
kernel matrices:

    MMD^2 = mean(k(x,x)) + mean(k(y,y)) - 2 * mean(k(x,y))

The median heuristic is computed under ``torch.no_grad()`` so the bandwidth is
a constant *scale*, not something the optimizer can shrink to game the loss.
"""
from __future__ import annotations

from typing import Sequence

import torch

DEFAULT_KERNEL_MULTIPLIERS = (0.5, 1.0, 2.0)


def _sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2.0) ** 2


def median_sq_distance(combined: torch.Tensor) -> torch.Tensor:
    """Median of pairwise squared distances, excluding the zero diagonal."""
    with torch.no_grad():
        d2 = _sq_dists(combined, combined)
        n = d2.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=d2.device)
        median = d2[mask].median()
    return median.clamp_min(1e-8)


def mmd2_multikernel(
    source: torch.Tensor,
    target: torch.Tensor,
    multipliers: Sequence[float] = DEFAULT_KERNEL_MULTIPLIERS,
) -> torch.Tensor:
    """Squared MMD between two feature batches with multi-kernel RBF."""
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("MMD needs non-empty source and target batches")
    combined = torch.cat([source, target], dim=0)
    median = median_sq_distance(combined)

    k_xx_raw = _sq_dists(source, source)
    k_yy_raw = _sq_dists(target, target)
    k_xy_raw = _sq_dists(source, target)

    loss = source.new_zeros(())
    for multiplier in multipliers:
        sigma2 = multiplier * median
        k_xx = torch.exp(-k_xx_raw / (2.0 * sigma2)).mean()
        k_yy = torch.exp(-k_yy_raw / (2.0 * sigma2)).mean()
        k_xy = torch.exp(-k_xy_raw / (2.0 * sigma2)).mean()
        loss = loss + (k_xx + k_yy - 2.0 * k_xy)
    return loss
