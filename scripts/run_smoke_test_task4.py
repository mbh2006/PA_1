"""End-to-end smoke test of the Task 4 pipeline on CPU (small sample limits).

Downloads CIFAR-10/CIFAR-100 if needed, trains a few steps of Vanilla, GCSC and
PROSER, caches their outputs on a small subset, and runs the OSR evaluation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMMON = ["--smoke", "--steps-per-epoch", "3", "--max-epochs", "1",
          "--out-root", "results_smoke", "--ckpt-root", "checkpoints_smoke", "--device", "cpu"]


def run_module(arguments) -> None:
    command = [sys.executable, "-m"] + list(arguments)
    print(">>", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    os.chdir(ROOT)

    run_module(["task4.train", "--config", "task4/configs/vanilla.yaml",
                "--run-id", "smoke_t4_vanilla"] + COMMON)
    run_module(["task4.train", "--config", "task4/configs/gcsc.yaml",
                "--run-id", "smoke_t4_gcsc"] + COMMON)
    run_module(["task4.train", "--config", "task4/configs/proser.yaml",
                "--run-id", "smoke_t4_proser",
                "--init-ckpt", "checkpoints_smoke/smoke_t4_vanilla/best.pt"] + COMMON)

    for method in ["vanilla", "gcsc", "proser"]:
        run_module(["task4.extract_outputs", "--run-dir", f"results_smoke/smoke_t4_{method}",
                    "--limit", "192", "--batch-size", "64",
                    "--out", f"task4/cache_smoke/smoke_t4_{method}.npz", "--device", "cpu"])

    run_module(["task4.evaluate_osr",
                "--vanilla-run", "results_smoke/smoke_t4_vanilla",
                "--gcsc-run", "results_smoke/smoke_t4_gcsc",
                "--proser-run", "results_smoke/smoke_t4_proser",
                "--cache-dir", "task4/cache_smoke",
                "--out-dir", "results_smoke/t4_osr", "--data-root", "data"])

    expected = [
        ROOT / "results_smoke" / "smoke_t4_vanilla" / "metrics.json",
        ROOT / "results_smoke" / "smoke_t4_proser" / "metrics.json",
        ROOT / "results_smoke" / "t4_osr" / "osr_metrics.json",
        ROOT / "results_smoke" / "t4_osr" / "table_posthoc.csv",
        ROOT / "results_smoke" / "t4_osr" / "table_models.csv",
        ROOT / "results_smoke" / "t4_osr" / "roc_scores.png",
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                json.load(fh)

    with open(ROOT / "results_smoke" / "t4_osr" / "osr_metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert len(metrics["posthoc_vanilla"]) == 4
    assert len(metrics["models"]) == 4  # vanilla, gcsc, proser MLS + proser placeholder
    print("  [ok] Task 4 artifacts exist and parse")
    print("\nTASK 4 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
