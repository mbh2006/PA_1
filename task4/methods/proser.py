"""Task 4 Step 4: PROSER - classifier and data placeholders.

Implements the two objectives from Zhou et al. (2021), "Learning Placeholders
for Open-Set Recognition":

Classifier placeholders (Eq. 5), first half of each batch:

    l1 = CE(augmented_logits, y) + beta * CE(masked_probabilities, dummy)
    augmented_logits = [known logits, strongest dummy logit]
    masked_probabilities = softmax(augmented) with the true-class entry zeroed
    and renormalised, then the dummy entry is pushed to be the strongest of
    the remaining responses.

Data placeholders (Eq. 7), second half of each batch: manifold mixup after
layer2 / before layer3, mixing hidden features of two *different* classes
(lambda ~ Beta(2, 2)) and training the mixed representation toward the dummy
classifiers; these proxy unknowns tighten the known-class regions.

The total objective is l1 + gamma * l2. No CIFAR-100 image appears anywhere.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class PROSER:
    name = "proser"

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        proser_cfg = cfg.get("proser", {})
        self.beta = float(proser_cfg.get("beta", 1.0))
        self.gamma = float(proser_cfg.get("gamma", 0.1))

    # ---------------------------------------------------------------- pieces
    @staticmethod
    def classifier_placeholder_loss(augmented_logits, targets, beta):
        ce = F.cross_entropy(augmented_logits, targets)

        probabilities = F.softmax(augmented_logits, dim=1)
        mask = torch.zeros_like(probabilities).scatter_(1, targets.unsqueeze(1), 1.0)
        masked = probabilities * (1.0 - mask)
        masked = masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-12)
        dummy_probability = masked[:, -1].clamp_min(1e-12)
        placeholder = -torch.log(dummy_probability).mean()
        return ce + beta * placeholder

    @staticmethod
    def data_placeholder_loss(model, mixed_features, valid_pairs):
        logits = model.classifier(mixed_features)
        augmented = model.augmented_logits(logits)
        dummy_target = torch.full((augmented.size(0),), model.num_known,
                                  dtype=torch.long, device=augmented.device)
        per_sample = F.cross_entropy(augmented, dummy_target, reduction="none")
        valid = valid_pairs.float()
        return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ loss
    def compute_loss(self, model, images, labels, progress):
        half = images.size(0) // 2
        first_images, first_labels = images[:half], labels[:half]
        second_images, second_labels = images[half:], labels[half:]

        # ---- classifier placeholders on the first half
        logits, _ = model(first_images)
        l1 = self.classifier_placeholder_loss(model.augmented_logits(logits),
                                              first_labels, self.beta)

        # ---- data placeholders on the second half (manifold mixup)
        hidden = model.features_pre(second_images)
        permutation = torch.randperm(hidden.size(0), device=hidden.device)
        hidden_partner = hidden[permutation]
        labels_partner = second_labels[permutation]

        lam = torch.distributions.Beta(
            torch.tensor(2.0, device=hidden.device),
            torch.tensor(2.0, device=hidden.device),
        ).sample((hidden.size(0),)).view(-1, 1, 1, 1)
        mixed = lam * hidden + (1.0 - lam) * hidden_partner
        mixed_features = model.features_post(mixed)
        l2 = self.data_placeholder_loss(model, mixed_features, second_labels != labels_partner)

        total = l1 + self.gamma * l2
        return total, {"cls_loss": l1.detach(), "mix_loss": l2.detach()}
