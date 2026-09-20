"""Push a list of Kaggle kernels respecting the 2-concurrent-GPU-session limit.

    python scripts/kaggle_queue.py t2_dan t3_dan_dg t2_dan_lambda01 ... 
        [--concurrency 2] [--interval 60]

The script pushes kernels as GPU slots free up, waits until all finish, prints
a final status table (and exits non-zero if any kernel ended in ERROR or
CANCELLED). Outputs are downloaded separately with
``python scripts/kaggle_push.py pull <name>``.
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

from scripts.kaggle_push import DONE_STATES, KERNELS_DIR, kaggle_cli, kernel_id  # noqa: E402


def state_of(name: str) -> str:
    completed = subprocess.run([kaggle_cli(), "kernels", "status", kernel_id(name)],
                               capture_output=True, text=True)
    text = (completed.stdout or "").strip()
    if '"' in text:
        return text.split('"')[1]
    return "UNKNOWN"


def push(name: str) -> bool:
    folder = KERNELS_DIR / name
    completed = subprocess.run([kaggle_cli(), "kernels", "push", "-p", str(folder)],
                               capture_output=True, text=True)
    output = (completed.stdout or "") + (completed.stderr or "")
    if "successfully pushed" in output:
        print(f"  pushed {name}", flush=True)
        return True
    if "Maximum batch GPU session count" in output:
        return False
    print(f"  PUSH FAILED for {name}:\n{output[-500:]}", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="kernel folder names under kaggle/kernels")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--timeout-min", type=int, default=300)
    args = parser.parse_args()

    pending = list(args.names)
    active: dict[str, str] = {}
    finished: dict[str, str] = {}
    deadline = time.time() + args.timeout_min * 60

    while (pending or active) and time.time() < deadline:
        while pending and len(active) < args.concurrency:
            name = pending[0]
            if push(name):
                pending.pop(0)
                active[name] = "PUSHED"
            else:
                break  # no slot yet; wait
            time.sleep(5)

        for name in list(active):
            state = state_of(name)
            active[name] = state
            if state in DONE_STATES:
                finished[name] = state
                del active[name]

        stamp = datetime.now().strftime("%H:%M:%S")
        summary = "  ".join(f"{k}={v.replace('KernelWorkerStatus.', '')}" for k, v in active.items())
        print(f"{stamp}  pending={len(pending)}  active: {summary or '-'}", flush=True)
        if pending or active:
            time.sleep(args.interval)

    print("\n=== final statuses ===")
    failed = []
    for name in args.names:
        state = finished.get(name) or state_of(name)
        print(f"  {name:22s} {state}")
        if state != "KernelWorkerStatus.COMPLETE":
            failed.append(name)
    if failed:
        sys.exit(f"kernels not complete: {failed}")
    print("ALL QUEUED KERNELS COMPLETE")


if __name__ == "__main__":
    main()
