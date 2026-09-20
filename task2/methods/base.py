"""Common interface for Task 2 methods.

The training loop only knows about ``Method.compute``: it receives the model,
one domain-balanced batch and the training progress in [0, 1], and returns the
total loss plus a dict of scalars for logging. Allowed information is made
explicit through ``requires_target``: Source-only must never see Sketch.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Method(nn.Module):
    name = "base"
    requires_target = False

    def __init__(self, cfg: Dict, device: torch.device, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    # ---------------------------------------------------------------- helpers
    def extra_parameters(self) -> List[torch.nn.Parameter]:
        """Parameters outside the backbone (e.g. the domain discriminator)."""
        return []

    @staticmethod
    def source_classification(model, batch: Dict[str, torch.Tensor]):
        logits, features = model(batch["source_x"])
        loss = F.cross_entropy(logits, batch["source_y"])
        return logits, features, loss

    # ------------------------------------------------------------- interface
    def compute(self, model, batch: Dict[str, torch.Tensor], progress: float):
        raise NotImplementedError

    def extra_state_dict(self) -> Dict[str, object]:
        return {}
