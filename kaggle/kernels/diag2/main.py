"""GPU diagnostic: does the Task 1 dependency install sequence survive a GPU worker?

CPU-only workers worked; this checks the GPU image with the exact command order
used by the Task 1 kernel. Mirrors progress to /kaggle/working/STATUS.txt.
"""
from __future__ import annotations

import subprocess
import sys
import traceback

STATUS = "/kaggle/working/STATUS.txt"


def log(line: str) -> None:
    with open(STATUS, "w", encoding="utf-8") as fh:
        fh.write(str(line) + "\n")
    print(line, flush=True)


def run(label: str, args) -> None:
    log("running: " + label)
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        log(f"FAILED ({label}) rc={completed.returncode}\n"
            + (completed.stderr or completed.stdout)[-1500:])
        raise SystemExit(1)
    log("ok: " + label)


def main() -> None:
    try:
        subprocess.run(["nvidia-smi", "-L"])
        run("python version check", [sys.executable, "-c", "import sys; print(sys.version)"])
        run("pip pyyaml scikit-learn tqdm",
            [sys.executable, "-m", "pip", "install", "-q", "pyyaml", "scikit-learn", "tqdm"])
        run("pip open_clip --no-deps",
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "open_clip_torch"])
        run("pip pure-python deps",
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "ftfy", "regex", "timm", "safetensors", "huggingface_hub"])
        run("import open_clip",
            [sys.executable, "-c",
             "import open_clip; print('open_clip', open_clip.__version__)"])
        log("GPU DIAG DONE")
    except Exception:
        with open(STATUS, "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
