"""Diagnostic kernel: pin down the working open_clip install recipe on Kaggle.

CPU-only (no GPU quota). Appends everything to /kaggle/working/STATUS.txt so the
result survives even when the log stream is empty.
"""
from __future__ import annotations

import glob
import subprocess
import sys
import traceback

STATUS = "/kaggle/working/STATUS.txt"


def log(line: str) -> None:
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(str(line) + "\n")
    print(line, flush=True)


def check(label: str, args, timeout: int = 1200) -> bool:
    completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    log(f"[{label}] rc={completed.returncode}")
    if completed.stdout.strip():
        log("  stdout: " + completed.stdout.strip()[-1200:])
    if completed.stderr.strip():
        log("  stderr: " + completed.stderr.strip()[-1500:])
    return completed.returncode == 0


def main() -> None:
    open(STATUS, "w").close()
    try:
        check("pip open_clip --no-deps",
              [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "open_clip_torch"])
        import_code = ("import open_clip, torch, torchvision; "
                       "print('versions', open_clip.__version__, torch.__version__, torchvision.__version__)")
        if not check("import 1", [sys.executable, "-c", import_code]):
            check("pip pure-python deps",
                  [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                   "ftfy", "regex", "timm", "safetensors", "huggingface_hub"])
            check("import 2", [sys.executable, "-c", import_code])

        log("mounts under /kaggle/input:")
        for path in sorted(glob.glob("/kaggle/input/*")):
            log("  " + path)
        for pattern in ["/kaggle/input/*/best.pt", "/kaggle/input/*/*/best.pt",
                        "/kaggle/input/*/t4_vanilla.npz", "/kaggle/input/*/*/t4_vanilla.npz"]:
            for candidate in glob.glob(pattern):
                log("  asset: " + candidate)
        log("DIAG DONE")
    except Exception:
        with open(STATUS, "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
