"""Task 3 Step 2: DAN-DG - pairwise source-domain alignment.

    L_DAN-DG = L_ERM + (lambda_DG / 3) * sum_{e<e'} MMD^2(F(X_e), F(X_e'))

The MMD mechanism is identical to Task 2's DAN (same implementation, same
kernel construction per batch), but it is applied to pairs of *observed source
domains*. Sketch is never accessed; this is what makes it a domain
generalisation method rather than a domain adaptation method.
"""
from __future__ import annotations

import torch.nn.functional as F

from shared.mmd import DEFAULT_KERNEL_MULTIPLIERS, mmd2_multikernel
from shared.pacs import SOURCE_DOMAINS
from task3.methods.erm import ERM


class DANDG(ERM):
    name = "dan_dg"

    def __init__(self, cfg, device, num_classes):
        super().__init__(cfg, device, num_classes)
        dan_cfg = cfg.get("dan_dg", {})
        self.lambda_dg = float(dan_cfg.get("lambda_dg", 1.0))
        self.kernels = tuple(dan_cfg.get("kernels", DEFAULT_KERNEL_MULTIPLIERS))

    def compute_loss(self, model, batch, progress):
        logits, features = model(batch["source_x"])
        cls_loss = F.cross_entropy(logits, batch["source_y"])

        # The domain-balanced iterator concatenates the source domains in order,
        # so features can be sliced back into per-domain groups.
        per_domain = self.cfg["batch_size_per_domain"]
        groups = [
            features[i * per_domain:(i + 1) * per_domain]
            for i in range(len(SOURCE_DOMAINS))
        ]

        total_mmd = features.new_zeros(())
        pairs = 0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                total_mmd = total_mmd + mmd2_multikernel(groups[i], groups[j], self.kernels)
                pairs += 1
        mean_mmd = total_mmd / max(1, pairs)

        loss = cls_loss + self.lambda_dg * mean_mmd
        return loss, {"cls_loss": cls_loss.detach(), "mmd": mean_mmd.detach()}
