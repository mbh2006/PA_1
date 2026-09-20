"""Method registry: one training loop, four method-specific objectives."""
from __future__ import annotations

from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly

METHODS = {
    SourceOnly.name: SourceOnly,
    DAN.name: DAN,
    DANN.name: DANN,
    CDAN.name: CDAN,
}


def build_method(cfg, device, feature_dim: int, num_classes: int):
    name = cfg["method"]
    if name not in METHODS:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(METHODS)}")
    return METHODS[name](cfg, device, feature_dim, num_classes)
