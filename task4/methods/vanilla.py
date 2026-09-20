"""Task 4 Step 1: vanilla closed-set training (cross-entropy)."""
from __future__ import annotations

import torch.nn.functional as F


class Vanilla:
    name = "vanilla"

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def compute_loss(self, model, images, labels, progress):
        logits, _ = model(images)
        loss = F.cross_entropy(model.known_logits(logits), labels)
        return loss, {"cls_loss": loss.detach()}
