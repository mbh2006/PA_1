"""Step 3 of Task 2: DANN - adversarial alignment with a gradient-reversal layer.

A binary discriminator (256 -> ReLU -> dropout 0.5 -> 2) tries to separate
source from target features, while the gradient-reversal layer sends the
opposite gradient into the backbone:

    L = L_cls + L_domain

Only source examples contribute to L_cls; both source and target contribute to
L_domain. The reversal strength follows alpha(p) = 2 / (1 + exp(-10 p)) - 1,
where p is training progress (the controlled study varies max_alpha).

This method uses unlabeled Sketch images during adaptation (target-aware).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from shared.domain_discriminator import DomainDiscriminator
from shared.grl import grad_reverse, grl_alpha
from task2.methods.base import Method


class DANN(Method):
    name = "dann"
    requires_target = True

    def __init__(self, cfg, device, feature_dim, num_classes):
        super().__init__(cfg, device, feature_dim, num_classes)
        dann_cfg = cfg.get("dann", {})
        self.hidden_dim = int(dann_cfg.get("hidden_dim", 256))
        self.dropout = float(dann_cfg.get("dropout", 0.5))
        self.max_alpha = float(dann_cfg.get("max_alpha", 1.0))
        self.discriminator = DomainDiscriminator(
            in_dim=feature_dim, hidden_dim=self.hidden_dim, dropout=self.dropout
        ).to(device)

    def extra_parameters(self):
        return list(self.discriminator.parameters())

    def compute(self, model, batch, progress):
        logits, features_s, cls_loss = self.source_classification(model, batch)
        _, features_t = model(batch["target_x"])

        alpha = grl_alpha(progress, max_alpha=self.max_alpha)
        reversed_features = torch.cat(
            [grad_reverse(features_s, alpha), grad_reverse(features_t, alpha)], dim=0
        )
        domain_labels = torch.cat(
            [
                torch.zeros(features_s.size(0), dtype=torch.long, device=features_s.device),
                torch.ones(features_t.size(0), dtype=torch.long, device=features_t.device),
            ]
        )
        domain_logits = self.discriminator(reversed_features)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        total = cls_loss + domain_loss
        with torch.no_grad():
            domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean()
        return total, {
            "cls_loss": cls_loss.detach(),
            "dom_loss": domain_loss.detach(),
            "dom_acc": domain_acc.detach(),
            "alpha": alpha,
        }

    def extra_state_dict(self):
        return {"discriminator": self.discriminator.state_dict()}
