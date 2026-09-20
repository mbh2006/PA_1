"""GPU diagnostic: accessing the mounted STL-10 dataset files.

Previous runs died the moment the mounted dataset was first touched. This
isolates directory listing, file stat/read and the torchvision load, logging
each step to /kaggle/working/STATUS.txt.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import traceback

STATUS = "/kaggle/working/STATUS.txt"


def log(line: str) -> None:
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(str(line) + "\n")
    print(line, flush=True)


def main() -> None:
    open(STATUS, "w").close()
    try:
        log("listing /kaggle/input")
        for root, dirs, files in os.walk("/kaggle/input"):
            depth = root.count("/")
            if depth > 5:
                dirs[:] = []
                continue
            log(f"  {root}  dirs={len(dirs)} files={len(files)}")
            dirs.sort()

        matches = glob.glob("/kaggle/input/*/*/*/stl10_binary") + \
            glob.glob("/kaggle/input/*/*/stl10_binary") + glob.glob("/kaggle/input/*/stl10_binary")
        log(f"glob matches: {matches}")
        if not matches:
            log("no stl10_binary found")
            return
        stl = matches[0]

        log("ls -la " + stl)
        completed = subprocess.run(["ls", "-la", stl], capture_output=True, text=True)
        log(completed.stdout[-1200:])

        train_x = os.path.join(stl, "train_X.bin")
        log(f"train_X.bin size: {os.path.getsize(train_x)}")
        with open(train_x, "rb") as fh:
            block = fh.read(1_000_000)
        log(f"read {len(block)} bytes of train_X.bin")

        log("torchvision STL10 load")
        from torchvision.datasets import STL10
        root = os.path.dirname(stl)
        train = STL10(root=root, split="train", download=False)
        test = STL10(root=root, split="test", download=False)
        log(f"STL10 train={len(train)} test={len(test)} classes={train.classes}")
        image, label = train[0]
        log(f"first sample: size={image.size} label={label}")
        log("DIAG4 DONE")
    except Exception:
        with open(STATUS, "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
