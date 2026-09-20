"""BatchNorm policy required by Tasks 2 and 3.

The assignment freezes the BatchNorm *running statistics* at their pretrained
ImageNet values for every method, because updating them on source/target
mixtures would add an implicit, uncontrolled form of adaptation. The affine
parameters gamma and beta must remain trainable.

PyTorch detail: calling ``model.train()`` puts BN modules back into training
mode (which updates running stats). After calling ``model.train()`` once, we
place only the BatchNorm modules in eval mode. Their running_mean / running_var
then stop updating, while gamma/beta still receive gradients because they are
ordinary parameters.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

BN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def freeze_bn_running_stats(model: nn.Module) -> int:
    """Put every BatchNorm module in eval mode; returns how many were touched."""
    count = 0
    for module in model.modules():
        if isinstance(module, BN_TYPES):
            module.eval()
            count += 1
    return count


@torch.no_grad()
def bn_snapshot(model: nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Snapshot of BN buffers, used by tests to prove they never change."""
    snap = {}
    for name, module in model.named_modules():
        if isinstance(module, BN_TYPES):
            snap[name] = (
                module.running_mean.detach().clone(),
                module.running_var.detach().clone(),
                module.num_batches_tracked.detach().clone(),
            )
    return snap


def assert_bn_unchanged(before: Dict, after: Dict, atol: float = 0.0) -> None:
    """Raise AssertionError if any BN running statistic changed."""
    assert before.keys() == after.keys(), "BN module set changed"
    for name in before:
        for i, (b, a) in enumerate(zip(before[name], after[name])):
            if not torch.equal(b, a):
                if atol == 0.0 or (b - a).abs().max().item() > atol:
                    raise AssertionError(
                        f"BatchNorm buffer {i} of '{name}' changed although it must stay frozen."
                    )
