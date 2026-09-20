"""Unattended safety net: wait for Kaggle kernels, then pull their outputs.

    python scripts/kaggle_autopull.py t4_rest t1_task1

Polls the given kernels until each reaches a terminal state (COMPLETE / ERROR /
CANCELLED), then downloads its output into ``kaggle_outputs/<name>/`` (retrying
transient download failures). Prints a final status summary and exits non-zero
if any kernel ended in ERROR/CANCELLED, so an attached watcher reports it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.kaggle_push import DONE_STATES, OUTPUTS_DIR, kaggle_cli, kernel_id  # noqa: E402


def state_of(name: str) -> str:
    completed = subprocess.run([kaggle_cli(), "kernels", "status", kernel_id(name)],
                               capture_output=True, text=True)
    text = (completed.stdout or "").strip()
    return text.split('"')[1] if '"' in text else "UNKNOWN"


def pull(name: str, retries: int = 3) -> bool:
    target = OUTPUTS_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    from scripts.kaggle_push import kaggle_env
    for attempt in range(1, retries + 1):
        completed = subprocess.run(
            [kaggle_cli(), "kernels", "output", kernel_id(name), "-p", str(target)],
            capture_output=True, encoding="utf-8", errors="replace", env=kaggle_env())
        if completed.returncode == 0:
            print(f"  pulled {name} -> {target}", flush=True)
            return True
        message = ((completed.stderr or "") + (completed.stdout or "")).strip()[-200:]
        print(f"  pull attempt {attempt} failed for {name}: {message}", flush=True)
        time.sleep(30)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--timeout-min", type=int, default=300)
    args = parser.parse_args()

    deadline = time.time() + args.timeout_min * 60
    states: dict[str, str] = {name: "UNKNOWN" for name in args.names}
    while time.time() < deadline and not all(s in DONE_STATES for s in states.values()):
        for name in args.names:
            if states[name] not in DONE_STATES:
                states[name] = state_of(name)
        stamp = datetime.now().strftime("%H:%M:%S")
        summary = "  ".join(f"{n}={s.replace('KernelWorkerStatus.', '')}" for n, s in states.items())
        print(f"{stamp}  {summary}", flush=True)
        if all(s in DONE_STATES for s in states.values()):
            break
        time.sleep(args.interval)

    print("\n=== pulling outputs ===", flush=True)
    failed: list[str] = []
    for name in args.names:
        if not pull(name):
            failed.append(name)

    print("\n=== final ===")
    for name in args.names:
        print(f"  {name:14s} {states[name]}")
    if failed:
        sys.exit(f"could not pull outputs for: {failed}")
    if any(s != "KernelWorkerStatus.COMPLETE" for s in states.values()):
        sys.exit("one or more kernels did not complete")
    print("ALL KERNELS COMPLETE AND OUTPUTS PULLED")


if __name__ == "__main__":
    main()
