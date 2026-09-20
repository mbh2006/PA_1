"""GPU diagnostic with the STL-10 dataset attached.

Tests whether attaching the 2.6 GB dataset plus the dependency install kills the
session, and records disk usage at each step (the prime suspect).
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback

STATUS = "/kaggle/working/STATUS.txt"


def log(line: str) -> None:
    with open(STATUS, "w", encoding="utf-8") as fh:
        fh.write(str(line) + "\n")
    print(line, flush=True)


def disk(label: str) -> None:
    completed = subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True, text=True)
    log(f"disk {label}: " + completed.stdout.strip().replace("\n", " | "))


def run(label: str, args) -> None:
    log("running: " + label)
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        log(f"FAILED ({label}) rc={completed.returncode}\n" + (completed.stderr or "")[-1200:])
        raise SystemExit(1)
    log("ok: " + label)


def main() -> None:
    try:
        subprocess.run(["nvidia-smi", "-L"])
        disk("start")
        log("mounts: " + " | ".join(sorted(os.listdir("/kaggle/input"))))
        for root, dirs, _ in os.walk("/kaggle/input"):
            depth = root.count("/")
            if depth > 4:
                dirs[:] = []
                continue
            if "stl10_binary" in dirs:
                log("stl10_binary at: " + root)
        run("pip pyyaml scikit-learn tqdm",
            [sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-learn", "tqdm"])
        disk("after pip 1")
        run("pip open_clip --no-deps",
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "open_clip_torch"])
        run("pip pure-python deps",
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "ftfy", "regex", "timm", "safetensors", "huggingface_hub"])
        run("import open_clip",
            [sys.executable, "-c", "import open_clip; print(open_clip.__version__)"])
        disk("after pip all")
        log("DIAG3 DONE")
    except Exception:
        with open(STATUS, "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
