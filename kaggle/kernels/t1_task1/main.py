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
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print("STATUS:", message, flush=True)


def run(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_captured(label, command, timeout=None):
    """Run a command, mirroring its combined output into STATUS."""
    log_status("run: " + label)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    tail = ((completed.stdout or "") + (completed.stderr or "")).strip()[-2000:]
    if tail:
        log_status(f"{label} rc={completed.returncode} output tail:\n{tail}")
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with rc={completed.returncode}")
    return completed


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
    subprocess.run(["df", "-h"])
    if not os.path.isdir(REPO_DIR):
        run_captured("git clone", ["git", "clone", REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    run_captured("pip base", [sys.executable, "-m", "pip", "install", "-q",
                              "pyyaml", "scikit-learn", "tqdm"])
    # Validated on Kaggle CPU and GPU workers: installing open_clip with its
    # dependencies can pull a new torch/CUDA stack and kill the session.
    run_captured("pip open_clip", [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                                   "open_clip_torch"])
    run_captured("pip open_clip deps", [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                                        "ftfy", "regex", "timm", "safetensors", "huggingface_hub"])
    run_captured("import open_clip", [sys.executable, "-c",
                                      "import open_clip; print('open_clip', open_clip.__version__)"])

    # Read STL-10 directly from the read-only dataset mount: copying the 2.6 GB
    # folder into the working disk is what killed earlier sessions (hard SIGKILL
    # with no traceback when the working disk filled up).
    stl = find_dir("stl10_binary", "train_X.bin")
    data_root = os.path.dirname(stl)
    log_status("starting pipeline | data_root=" + data_root)

    log_path = "/kaggle/working/pipeline.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        completed = subprocess.run(
            [sys.executable, "-u", "-m", "task1.run_task1", "--config", "task1/configs/base.yaml",
             "--stage", "all", "--device", "cuda", "--batch-size", "32", "--data-root", data_root],
            stdout=fh, stderr=subprocess.STDOUT, timeout=None)
    log_status(f"pipeline rc={completed.returncode}")
    if completed.returncode != 0:
        raise SystemExit(f"task1 pipeline failed rc={completed.returncode} (see pipeline.log)")

    # keep the frozen subsets and the cue-conflict manifest in the output
    for relative in ("task1/data/subsets_stl10_seed6304.json",
                     "task1/data/cue_conflicts/manifest.json"):
        source = os.path.join(REPO_DIR, relative)
        if os.path.exists(source):
            shutil.copy2(source, "/kaggle/working/" + os.path.basename(relative))
    shutil.make_archive("/kaggle/working/results_task1", "zip", "results", "task1")
    log_status("DONE")


def main():
    try:
        pipeline()
    except Exception:
        with open(STATUS, "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
