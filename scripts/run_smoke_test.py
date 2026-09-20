"""End-to-end smoke test of the Task 2 pipeline on CPU, using fake data.

It verifies the things that are easy to get silently wrong and expensive to
debug on Kaggle:

1. fake PACS tree + deterministic seed-6304 split file;
2. batch composition: exactly 8 examples per source domain (+24 target);
3. BatchNorm running statistics never change while gamma/beta still get grads;
4. all four methods (source_only, dan, dann, cdan) train, evaluate and write
   their result files;
5. final evaluation + domain separability run on a finished checkpoint.

    python scripts/run_smoke_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from common.config import load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402
from shared.backbone import ResNet18PACS  # noqa: E402
from shared.bn_policy import assert_bn_unchanged, bn_snapshot, freeze_bn_running_stats  # noqa: E402
from shared.pacs import SOURCE_DOMAINS  # noqa: E402
from shared.pacs_protocol import build_splits, load_splits, save_splits  # noqa: E402
from task2.methods import build_method  # noqa: E402
from task2.train import build_iterator, to_device  # noqa: E402

METHODS = ["source_only", "dan", "dann", "cdan"]


def run_module(arguments) -> None:
    command = [sys.executable, "-m"] + list(arguments)
    print(">>", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def check_batch_and_bn(data_root: Path, splits_path: Path) -> None:
    cfg = load_config(ROOT / "task2/configs/dann.yaml")
    cfg["num_workers"] = 0
    device = torch.device("cpu")
    set_seed(cfg["seed"])

    model = ResNet18PACS().to(device)
    method = build_method(cfg, device, model.feature_dim, cfg["num_classes"])
    splits = load_splits(splits_path)
    iterator = build_iterator(cfg, data_root, splits, method, pin_memory=False)

    batch = iterator.next_batch()
    assert batch["source_x"].shape[0] == 24, batch["source_x"].shape
    domain_ids = batch["source_domain"].numpy()
    counts = {d: int((domain_ids == i).sum()) for i, d in enumerate(SOURCE_DOMAINS)}
    assert all(count == 8 for count in counts.values()), counts
    assert batch["target_x"].shape[0] == 24, batch["target_x"].shape
    print(f"  [ok] batch composition {counts} + 24 target")

    model.train()
    freeze_bn_running_stats(model)
    before = bn_snapshot(model)
    optimizer = AdamW(list(model.parameters()) + list(method.extra_parameters()), lr=1e-4)
    loss, _ = method.compute(model, to_device(batch, device), progress=0.5)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = bn_snapshot(model)
    assert_bn_unchanged(before, after)
    bn_grad = model.net.bn1.weight.grad
    assert bn_grad is not None and torch.isfinite(bn_grad).all(), "BN affine params got no gradient"
    print("  [ok] BN running stats frozen; BN affine parameters still trained")


def check_artifacts() -> None:
    expected = [
        ROOT / "results_smoke" / "smoke_source_only" / "metrics.json",
        ROOT / "results_smoke" / "smoke_dan" / "metrics.json",
        ROOT / "results_smoke" / "smoke_dann" / "metrics.json",
        ROOT / "results_smoke" / "smoke_cdan" / "metrics.json",
        ROOT / "results_smoke" / "smoke_dan" / "final_metrics.json",
        ROOT / "results_smoke" / "smoke_dan" / "domain_separability.json",
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                json.load(fh)
    print("  [ok] all expected result files exist and parse")


def main() -> None:
    os.chdir(ROOT)
    data_root = ROOT / "data" / "fake_pacs"
    splits_path = data_root / "splits_seed6304.json"

    from scripts.make_fake_pacs import make_fake_pacs
    make_fake_pacs(data_root)
    splits = build_splits(data_root, seed=6304)
    save_splits(splits, splits_path)
    print(f"  [ok] fake PACS + split file written to {splits_path}")

    check_batch_and_bn(data_root, splits_path)

    for method in METHODS:
        run_module([
            "task2.train",
            "--config", f"task2/configs/{method}.yaml",
            "--data-root", str(data_root),
            "--splits", str(splits_path),
            "--smoke",
            "--steps-per-epoch", "3",
            "--run-id", f"smoke_{method}",
            "--out-root", "results_smoke",
            "--ckpt-root", "checkpoints_smoke",
            "--device", "cpu",
        ])

    run_module(["task2.evaluate_final", "--run-dir", "results_smoke/smoke_dan"])
    run_module(["task2.evaluation.domain_separability", "--run-dir", "results_smoke/smoke_dan"])
    check_artifacts()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
