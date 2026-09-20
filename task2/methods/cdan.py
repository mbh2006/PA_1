"""Step 4 of Task 2: CDAN - class-conditional adversarial alignment.

Instead of the bare feature, the domain discriminator sees the outer product of
the feature and the classifier's probability vector:

    g(x) = vec(f (x) p)     with p = softmax(C(f(x)))

This conditions alignment on the model's belief about the class, so examples
from the same class but different domains can be pulled together. The
assignment forbids entropy conditioning and requires that neither f nor p is
detached, so gradients flow through both.

Discriminator architecture, dropout, reversal schedule and loss weight are
identical to DANN; only the input changes (512 * 7 = 3584 dimensions).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from shared.domain_discriminator import DomainDiscriminator
from shared.grl import grad_reverse, grl_alpha
from task2.methods.base import Method


class CDAN(Method):
    name = "cdan"
    requires_target = True

    def __init__(self, cfg, device, feature_dim, num_classes):
        super().__init__(cfg, device, feature_dim, num_classes)
        cdan_cfg = cfg.get("cdan", {})
        self.hidden_dim = int(cdan_cfg.get("hidden_dim", 256))
        self.dropout = float(cdan_cfg.get("dropout", 0.5))
        self.max_alpha = float(cdan_cfg.get("max_alpha", 1.0))
        # See DANN for why the feature entering the domain head is normalised.
        self.normalize_features = bool(cdan_cfg.get("normalize_features", True))
        self.discriminator = DomainDiscriminator(
            in_dim=feature_dim * num_classes, hidden_dim=self.hidden_dim, dropout=self.dropout
        ).to(device)

    def extra_parameters(self):
        return list(self.discriminator.parameters())

    @staticmethod
    def _conditioned(features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """vec(f (x) p): batch outer product of features and softmax probabilities."""
        probabilities = F.softmax(logits, dim=1)
        return torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1)).flatten(1)

    def compute(self, model, batch, progress):
        logits_s, features_s, cls_loss = self.source_classification(model, batch)
        logits_t, features_t = model(batch["target_x"])

        if self.normalize_features:
            features_s = F.normalize(features_s, dim=1)
            features_t = F.normalize(features_t, dim=1)

        g_s = self._conditioned(features_s, logits_s)
        g_t = self._conditioned(features_t, logits_t)

        alpha = grl_alpha(progress, max_alpha=self.max_alpha)
        reversed_features = torch.cat([grad_reverse(g_s, alpha), grad_reverse(g_t, alpha)], dim=0)
        domain_labels = torch.cat(
            [
                torch.zeros(g_s.size(0), dtype=torch.long, device=g_s.device),
                torch.ones(g_t.size(0), dtype=torch.long, device=g_t.device),
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
        }

    def extra_state_dict(self):
        return {"discriminator": self.discriminator.state_dict()}
