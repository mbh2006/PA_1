"""Loading a trained model from a run directory (Task 2 or Task 3 runs)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from shared.backbone import ResNet18PACS


def load_model_from_run(run_dir: str | Path, ckpt_path: str | None = None,
                        device: torch.device | None = None) -> Tuple[Dict, torch.nn.Module, Path]:
    """Rebuild the model recorded in ``run_dir/config.json`` and load its checkpoint.

    Works for both Task 2 run folders (e.g. the source-only ERM baseline) and
    Task 3 runs, because both save the same config keys and checkpoint format.
    """
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(run_dir / "config.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    checkpoint_path = Path(ckpt_path) if ckpt_path else (
        Path(cfg.get("ckpt_root", "checkpoints")) / run_dir.name / "best.pt"
    )
    model = ResNet18PACS(num_classes=cfg["num_classes"]).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return cfg, model, checkpoint_path
