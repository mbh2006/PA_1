"""Loading trained Task 4 models from a run directory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from task4.models.resnet_cifar import CifarResNet18


def load_task4_model(run_dir: str | Path, ckpt_path: str | None = None,
                     device: torch.device | None = None) -> Tuple[Dict, torch.nn.Module, Path]:
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(run_dir / "config.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    checkpoint_path = Path(ckpt_path) if ckpt_path else (
        Path(cfg.get("ckpt_root", "checkpoints")) / run_dir.name / "best.pt"
    )
    num_dummies = int(cfg.get("proser", {}).get("num_dummies", 0)) if cfg["method"] == "proser" else 0
    model = CifarResNet18(num_classes=cfg["num_classes"], num_dummies=num_dummies).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return cfg, model, checkpoint_path
