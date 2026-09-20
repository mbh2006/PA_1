"""Gradient reversal layer and the DANN alpha schedule.

DANN trains a domain discriminator on top of the 512-d feature. While the
discriminator minimizes its own classification loss, the backbone must *maximize*
it, so gradients flowing back through the reversal layer are negated and scaled
by alpha(p):

    alpha(p) = 2 / (1 + exp(-10 p)) - 1,      p = training progress in [0, 1]

Early in training alpha is small, so the representation first learns the class
task; later the reversed domain gradient grows towards 1.
"""
from __future__ import annotations

import math

import torch


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.alpha = alpha  # type: ignore[attr-defined]
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Identity in the forward pass, ``-alpha * grad`` in the backward pass."""
    return _GradientReversal.apply(x, alpha)


def grl_alpha(progress: float, max_alpha: float = 1.0, gamma: float = 10.0) -> float:
    """Standard DANN schedule; ``max_alpha`` lets the controlled study vary the
    maximum reversal strength over {0.25, 0.5, 1} while keeping the shape."""
    p = float(min(max(progress, 0.0), 1.0))
    return float(max_alpha * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0))
