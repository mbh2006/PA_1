"""Task 4 method registry (Vanilla, GCSC, PROSER)."""
from __future__ import annotations

from task4.methods.gcsc import GCSC
from task4.methods.proser import PROSER
from task4.methods.vanilla import Vanilla

METHODS = {Vanilla.name: Vanilla, GCSC.name: GCSC, PROSER.name: PROSER}


def build_method(cfg, device):
    name = cfg["method"]
    if name not in METHODS:
        raise KeyError(f"Unknown Task 4 method '{name}'. Available: {sorted(METHODS)}")
    return METHODS[name](cfg, device)
