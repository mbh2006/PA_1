"""Binary domain discriminator used by DANN and CDAN.

The assignment fixes the architecture for both methods:
256-unit hidden layer -> ReLU -> dropout 0.5 -> 2-class output.

* DANN feeds the 512-d backbone feature.
* CDAN feeds the outer product vec(f (x) p) where p is the classifier's
  probability vector, so the input dimension is 512 * num_classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
