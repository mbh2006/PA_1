"""Step 1 of Task 2: Source-only ERM.

Cross-entropy on the three labeled source domains. This checkpoint is also the
shared ERM baseline for Task 3, so it must be saved and reused unchanged.
"""
from __future__ import annotations

from task2.methods.base import Method


class SourceOnly(Method):
    name = "source_only"
    requires_target = False

    def compute(self, model, batch, progress):
        logits, features, cls_loss = self.source_classification(model, batch)
        return cls_loss, {"cls_loss": cls_loss.detach()}
