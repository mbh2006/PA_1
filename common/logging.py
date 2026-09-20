"""Run logging: every run gets a folder with its config, history and metrics.

Layout produced per run::

    results/<run_id>/config.json      # merged config + config hash
    results/<run_id>/history.csv      # one row per epoch
    results/<run_id>/metrics.json     # best epoch / final numbers
    checkpoints/<run_id>/best.pt      # checkpoint chosen on mean source-val macro-F1
    checkpoints/<run_id>/last.pt      # for --resume after a Kaggle session ends
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

RESULTS_ROOT = Path("results")
CHECKPOINT_ROOT = Path("checkpoints")


def _clean(obj: Any) -> Any:
    """Replace NaN/inf with null so the JSON is standards-compliant."""
    import math

    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def get_logger(name: str = "pa1") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def make_run_id(task: str, method: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{task}_{method}_seed{seed}_{stamp}"


class RunLogger:
    """Writes small, machine-readable result files (safe to commit to git)."""

    def __init__(
        self,
        run_id: str,
        results_root: str | Path = RESULTS_ROOT,
        checkpoint_root: str | Path = CHECKPOINT_ROOT,
    ) -> None:
        self.run_id = run_id
        self.results_dir = Path(results_root) / run_id
        self.ckpt_dir = Path(checkpoint_root) / run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- config
    def save_config(self, cfg: Dict[str, Any]) -> None:
        with open(self.results_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=True, default=str)

    # --------------------------------------------------------------- history
    def log_row(self, row: Dict[str, Any]) -> None:
        self._history.append(
            {
                k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "epoch" else v)
                for k, v in row.items()
            }
        )
        self._write_history()

    def _write_history(self) -> None:
        if not self._history:
            return
        keys: List[str] = []
        for row in self._history:
            for k in row:
                if k not in keys:
                    keys.append(k)
        # keep a stable, readable order: epoch first
        if "epoch" in keys:
            keys.remove("epoch")
            keys = ["epoch"] + keys
        with open(self.results_dir / "history.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in self._history:
                writer.writerow({k: row.get(k, "") for k in keys})

    # ----------------------------------------------------------------- files
    def save_json(self, name: str, obj: Any) -> Path:
        path = self.results_dir / name
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_clean(obj), fh, indent=2, default=str)
        return path

    def save_csv(self, name: str, rows: List[Dict[str, Any]]) -> Path:
        path = self.results_dir / name
        if rows:
            keys = list(rows[0].keys())
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
        return path
