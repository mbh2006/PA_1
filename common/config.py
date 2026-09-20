"""YAML config loading with one level of inheritance and a stable config hash.

Every run saves its (merged) config next to its results, so any reported number
can be traced back to the exact settings that produced it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a YAML config.

    If the file contains an ``inherit: base.yaml`` key, the parent file is
    loaded first and the current file is deep-merged on top of it.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_update(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Short stable hash of a config, useful as a run fingerprint."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]
