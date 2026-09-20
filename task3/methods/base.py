"""Task 3 method interface.

Unlike Task 2, no method here may use the target domain at all. A method owns
the whole optimisation step because SAM needs two forward/backward passes while
ERM and DAN-DG need one.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


class Task3Method:
    name = "base"

    def __init__(self, cfg: Dict, device: torch.device, num_classes: int) -> None:
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes

    def compute_loss(self, model, batch: Dict[str, torch.Tensor], progress: float):
        raise NotImplementedError

    def step(self, model, batch: Dict[str, torch.Tensor], optimizer, progress: float):
        """One optimisation step; returns (loss value for logging, scalar logs)."""
        loss, logs = self.compute_loss(model, batch, progress)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach()), logs
