"""Kaggle script-kernel: Task 1 inductive biases on STL-10.

Clones the repository, stages the STL-10 binary files from the attached Kaggle
dataset, then runs the full cached Task 1 pipeline: subset preparation, AdaIN
cue-conflict generation, feature caching for all three frozen backbones,
linear heads and the analysis (shape bias, translation curves, stability,
t-SNE). Results are zipped for a one-file download.

Progress and any failure traceback are mirrored to /kaggle/working/STATUS.txt
so the output is diagnosable even if the log stream is lost.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import traceback

REPO_URL = "https://github.com/mbh2006/PA_1.git"
REPO_DIR = "/kaggle/working/PA_1"
STATUS = "/kaggle/working/STATUS.txt"


def log_status(message):
    with open(STATUS, "w", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print("STATUS:", message, flush=True)


def run(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def find_dir(name, sentinel):
    patterns = [f"/kaggle/input/{{name}}", f"/kaggle/input/*/{{name}}",
                f"/kaggle/input/*/*/{{name}}", f"/kaggle/input/*/*/*/{{name}}"]
    for pattern in patterns:
        for candidate in glob.glob(pattern):
            if os.path.exists(os.path.join(candidate, sentinel)):
                return candidate
    raise SystemExit(f"dataset folder {{name}} (with {{sentinel}}) not found under /kaggle/input")


def pipeline():
    log_status("start")
    subprocess.run(["nvidia-smi", "-L"])
    if not os.path.isdir(REPO_DIR):
        run(["git", "clone", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    log_status("installing dependencies")
    run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-learn",
         "tqdm", "open_clip_torch"])

    stl = find_dir("stl10_binary", "train_X.bin")
    os.makedirs("data/stl10", exist_ok=True)
    shutil.copytree(stl, "data/stl10/stl10_binary", dirs_exist_ok=True)
    log_status("staged STL-10")

    run([sys.executable, "-m", "task1.run_task1", "--config", "task1/configs/base.yaml",
         "--stage", "all", "--device", "cuda", "--batch-size", "32"])
    log_status("task1 pipeline done")

    shutil.make_archive("/kaggle/working/results_task1", "zip", "results", "task1")
    log_status("DONE")


def main():
    try:
        pipeline()
    except Exception:
        with open(STATUS, "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
