"""Task 3 method registry: ERM, DAN-DG (source-pair MMD), SAM."""
from __future__ import annotations

from task3.methods.dan_dg import DANDG
from task3.methods.erm import ERM
from task3.methods.sam import SAM

METHODS = {ERM.name: ERM, DANDG.name: DANDG, SAM.name: SAM}


def build_method(cfg, device, num_classes):
    name = cfg["method"]
    if name not in METHODS:
        raise KeyError(f"Unknown Task 3 method '{name}'. Available: {sorted(METHODS)}")
    return METHODS[name](cfg, device, num_classes)
