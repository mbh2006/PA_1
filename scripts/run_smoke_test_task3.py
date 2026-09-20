"""End-to-end smoke test of the Task 3 pipeline on CPU and fake data.

Verifies that ERM-style and SAM-style steps run, that no Sketch data is needed
for training, and that the three Task 3 diagnostics (Sketch evaluation,
source-domain separability, sharpness) all produce parseable result files.

    python scripts/run_smoke_test_task3.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_module(arguments) -> None:
    command = [sys.executable, "-m"] + list(arguments)
    print(">>", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    os.chdir(ROOT)
    data_root = ROOT / "data" / "fake_pacs"
    splits_path = data_root / "splits_seed6304.json"
    assert splits_path.exists(), "run scripts/run_smoke_test.py first (it creates the fake data)"

    common = ["--data-root", str(data_root), "--splits", str(splits_path), "--smoke",
              "--steps-per-epoch", "3", "--out-root", "results_smoke",
              "--ckpt-root", "checkpoints_smoke", "--device", "cpu"]

    run_module(["task3.train", "--config", "task3/configs/dan_dg.yaml",
                "--run-id", "smoke_t3_dan_dg"] + common)
    run_module(["task3.train", "--config", "task3/configs/sam.yaml",
                "--run-id", "smoke_t3_sam"] + common)

    # the Task 2 smoke run must have final_metrics.json so the delta-vs-ERM path is exercised
    run_module(["task2.evaluate_final", "--run-dir", "results_smoke/smoke_source_only"])

    run_module(["task3.evaluate_sketch", "--model-run", "results_smoke/smoke_t3_sam",
                "--erm-run", "results_smoke/smoke_source_only"])
    run_module(["task3.evaluation.source_domain_separability",
                "--model-run", "results_smoke/smoke_t3_sam"])
    run_module(["task3.evaluation.sharpness", "--model-run", "results_smoke/smoke_t3_sam"])

    expected = [
        ROOT / "results_smoke" / "smoke_t3_dan_dg" / "metrics.json",
        ROOT / "results_smoke" / "smoke_t3_sam" / "metrics.json",
        ROOT / "results_smoke" / "smoke_t3_sam" / "final_metrics.json",
        ROOT / "results_smoke" / "smoke_t3_sam" / "source_domain_separability.json",
        ROOT / "results_smoke" / "smoke_t3_sam" / "sharpness.json",
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
    print("  [ok] Task 3 artifacts exist and parse")

    with open(ROOT / "results_smoke" / "smoke_t3_sam" / "sharpness.json", encoding="utf-8") as fh:
        sharpness = json.load(fh)
    assert sharpness["sharpness_delta"] >= 0.0, "sharpness delta must be non-negative by construction"
    print("  [ok] sharpness proxy computed:", round(sharpness["sharpness_delta"], 6))
    print("\nTASK 3 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
