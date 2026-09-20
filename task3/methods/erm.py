"""Task 3 Step 1: ERM on the three labelled source domains.

The Task 3 baseline is *not* retrained: it is the saved Task 2 source-only
checkpoint (same architecture, data, optimisation and seed), loaded unchanged.
This module exists so the same pipeline can also train ERM if needed for the
controlled study.
"""
from __future__ import annotations

import torch.nn.functional as F

from task3.methods.base import Task3Method


class ERM(Task3Method):
    name = "erm"

    def compute_loss(self, model, batch, progress):
        logits, _ = model(batch["source_x"])
        loss = F.cross_entropy(logits, batch["source_y"])
        return loss, {"cls_loss": loss.detach()}
