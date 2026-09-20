"""Step 2 of Task 2: DAN - MMD alignment of source and target features.

    L_DAN = L_cls + lambda_MMD * || mean(phi(F(x_s))) - mean(phi(F(x_t))) ||_H^2

The discrepancy is applied to the 512-d feature right before the classifier
head. Both the source *and* the target pass through the network with gradients,
because the backbone must be trained to reduce the discrepancy.

This method uses unlabeled Sketch images during adaptation (target-aware).
"""
from __future__ import annotations

from typing import Dict

from shared.mmd import DEFAULT_KERNEL_MULTIPLIERS, mmd2_multikernel
from task2.methods.base import Method


class DAN(Method):
    name = "dan"
    requires_target = True

    def __init__(self, cfg, device, feature_dim, num_classes):
        super().__init__(cfg, device, feature_dim, num_classes)
        dan_cfg = cfg.get("dan", {})
        self.lambda_mmd = float(dan_cfg.get("lambda_mmd", 1.0))
        self.kernels = tuple(dan_cfg.get("kernels", DEFAULT_KERNEL_MULTIPLIERS))

    def compute(self, model, batch, progress):
        logits, features_s, cls_loss = self.source_classification(model, batch)
        _, features_t = model(batch["target_x"])
        mmd = mmd2_multikernel(features_s, features_t, self.kernels)
        total = cls_loss + self.lambda_mmd * mmd
        return total, {
            "cls_loss": cls_loss.detach(),
            "mmd": mmd.detach(),
        }
