"""Kaggle script-kernel: Task 3 ERM baseline diagnostics (evaluation only).

Loads the Task 2 source-only checkpoint unchanged, then computes exactly the
same Task 3 diagnostics as the trained methods: source/destination metrics,
source-domain separability and the sharpness proxy.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/mbh2006/PA_1.git"
REPO_DIR = "/kaggle/working/PA_1"
ERM_RESULTS_DATASET = "/kaggle/input/pa1-t2-erm-results"
ERM_CKPT_DATASET = "/kaggle/input/pa1-t2-erm-ckpt"
RUN_ID = "t2_source_only"


def run(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-learn", "tqdm"])

    # assemble the Task 2 ERM run exactly as it was saved
    os.makedirs(os.path.join("results", RUN_ID), exist_ok=True)
    for filename in os.listdir(ERM_RESULTS_DATASET):
        shutil.copy2(os.path.join(ERM_RESULTS_DATASET, filename),
                     os.path.join("results", RUN_ID, filename))
    os.makedirs(os.path.join("checkpoints", RUN_ID), exist_ok=True)
    for filename in os.listdir(ERM_CKPT_DATASET):
        if filename.endswith(".pt"):
            shutil.copy2(os.path.join(ERM_CKPT_DATASET, filename),
                         os.path.join("checkpoints", RUN_ID, filename))

    run([sys.executable, "-m", "task3.evaluate_sketch", "--model-run",
         os.path.join("results", RUN_ID), "--out-dir", "results/t3_erm"])
    run([sys.executable, "-m", "task3.evaluation.source_domain_separability",
         "--model-run", os.path.join("results", RUN_ID), "--out-dir", "results/t3_erm"])
    run([sys.executable, "-m", "task3.evaluation.sharpness",
         "--model-run", os.path.join("results", RUN_ID), "--out-dir", "results/t3_erm"])
    shutil.make_archive("/kaggle/working/results_t3_erm", "zip", "results", "t3_erm")
    print("KERNEL DONE: t3_erm", flush=True)


if __name__ == "__main__":
    main()
