"""Task 3 Step 3: SAM - Sharpness-Aware Minimisation (non-adaptive).

    min_theta max_{||eps||_2 <= rho} L_ERM(theta + eps)

One step needs two forward/backward passes:
1. compute the loss and its gradient at theta, build the normalised ascent
   perturbation eps = rho * g / ||g||;
2. move to theta + eps, recompute the loss there and keep *that* gradient;
3. restore theta and apply the optimizer update.

The frozen-BatchNorm policy from Task 2 is used in both passes (the training
loop keeps BN modules in eval mode), so the two passes only differ in the
weights and the perturbation is a pure parameter-space quantity.
"""
from __future__ import annotations

import torch

from task3.methods.erm import ERM


class SAM(ERM):
    name = "sam"

    def __init__(self, cfg, device, num_classes):
        super().__init__(cfg, device, num_classes)
        self.rho = float(cfg.get("sam", {}).get("rho", 0.05))

    @torch.no_grad()
    def _ascent_perturbation(self, parameters, rho):
        grad_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in parameters]), 2
        )
        return [rho * p.grad.detach() / (grad_norm + 1e-12) for p in parameters]

    def step(self, model, batch, optimizer, progress):
        loss, logs = self.compute_loss(model, batch, progress)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        epsilon = self._ascent_perturbation(parameters, self.rho)

        with torch.no_grad():
            for parameter, eps in zip(parameters, epsilon):
                parameter.add_(eps)

        perturbed_loss, _ = self.compute_loss(model, batch, progress)
        optimizer.zero_grad(set_to_none=True)
        perturbed_loss.backward()

        with torch.no_grad():
            for parameter, eps in zip(parameters, epsilon):
                parameter.sub_(eps)

        optimizer.step()
        return float(perturbed_loss.detach()), logs
