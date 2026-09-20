"""End-to-end smoke test of the Task 1 pipeline on CPU with fake STL-10 data."""
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
    from scripts.make_fake_stl10 import make_fake_stl10
    make_fake_stl10()

    run_module(["task1.run_task1", "--config", "task1/configs/smoke.yaml", "--stage", "all",
                "--limit", "24", "--backbones", "resnet50", "--device", "cpu",
                "--batch-size", "8"])

    expected = [
        ROOT / "results_smoke" / "task1" / "analysis.json",
        ROOT / "results_smoke" / "task1" / "translation_curve.png",
        ROOT / "results_smoke" / "task1" / "heads" / "resnet50.pt",
    ]
    for path in expected:
        assert path.exists(), f"missing artifact: {path}"
    with open(ROOT / "results_smoke" / "task1" / "analysis.json", "r", encoding="utf-8") as fh:
        report = json.load(fh)
    assert "grayscale" in report["conditions"]
    assert "shuffle" in report["conditions"]
    assert "resnet50" in report["translation"]
    assert "resnet50" in report["conflicts"]
    print("  [ok] Task 1 artifacts exist and parse")
    print("\nTASK 1 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
