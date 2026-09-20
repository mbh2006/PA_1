"""Push / watch / pull PA-1 Kaggle kernels from the command line.

Usage (from the repo root)::

    python scripts/kaggle_push.py push  t2_source_only
    python scripts/kaggle_push.py watch t2_source_only
    python scripts/kaggle_push.py pull  t2_source_only
    python scripts/kaggle_push.py all   t2_source_only     # push + watch + pull

The Kaggle CLI must be installed in the active environment and the API token
saved at ``%USERPROFILE%\\.kaggle\\kaggle.json`` (Kaggle account -> Settings ->
Create New API Token).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = ROOT / "kaggle" / "kernels"
OUTPUTS_DIR = ROOT / "kaggle_outputs"
USER = "mbh2006"
DONE_STATES = {"KernelWorkerStatus.COMPLETE", "KernelWorkerStatus.ERROR",
               "KernelWorkerStatus.CANCELLED", "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"}

# The Kaggle CLI emits non-ASCII progress characters; cp1252 consoles crash on
# them, so force UTF-8 on this process and on every child.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def kaggle_env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    return env


def kaggle_cli() -> str:
    """Prefer the CLI next to the current interpreter (it may not be on PATH)."""
    candidate = Path(sys.executable).parent / ("Scripts/kaggle.exe" if os.name == "nt" else "bin/kaggle")
    return str(candidate) if candidate.exists() else "kaggle"


def kernel_id(name: str) -> str:
    with open(KERNELS_DIR / name / "kernel-metadata.json", "r", encoding="utf-8") as fh:
        return json.load(fh)["id"]


def push(name: str) -> None:
    folder = KERNELS_DIR / name
    if not folder.is_dir():
        sys.exit(f"kernel folder not found: {folder}")
    subprocess.run([kaggle_cli(), "kernels", "push", "-p", str(folder)],
                   check=True, env=kaggle_env())


def status(name: str) -> str:
    completed = subprocess.run([kaggle_cli(), "kernels", "status", kernel_id(name)],
                               capture_output=True, encoding="utf-8", errors="replace",
                               env=kaggle_env(), check=True)
    text = completed.stdout.strip()
    print(text)
    return text.split('"')[1] if '"' in text else text


def watch(name: str, interval: int = 30, timeout_min: int = 240) -> None:
    deadline = time.time() + timeout_min * 60
    state = ""
    while time.time() < deadline:
        state = status(name)
        if state in DONE_STATES:
            break
        time.sleep(interval)
    if state != "KernelWorkerStatus.COMPLETE":
        sys.exit(f"kernel {name} ended with state {state}")


def pull(name: str) -> Path:
    target = OUTPUTS_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run([kaggle_cli(), "kernels", "output", kernel_id(name), "-p", str(target)],
                   check=True, env=kaggle_env())
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["push", "watch", "pull", "all"])
    parser.add_argument("name", help="kernel folder name, e.g. t2_source_only")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--timeout-min", type=int, default=240)
    args = parser.parse_args()

    if args.action in ("push", "all"):
        push(args.name)
    if args.action in ("watch", "all"):
        watch(args.name, args.interval, args.timeout_min)
    if args.action in ("pull", "all"):
        print("outputs in", pull(args.name))


if __name__ == "__main__":
    main()
