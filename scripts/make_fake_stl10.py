"""Generate a tiny fake STL-10 tree (ImageFolder layout) for Task 1 smoke tests.

Layout matches the ``imagefolder`` backend in ``task1/data/dataset.py``::

    <out>/train/<class>/<index>.jpg
    <out>/test/<class>/<index>.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]


def make_fake_stl10(out: str | Path = "data/fake_stl10", train_per_class: int = 24,
                    test_per_class: int = 12, size: int = 96, seed: int = 6304) -> Path:
    out = Path(out)
    rng = np.random.RandomState(seed)
    colours = {name: rng.randint(40, 215, size=3).astype(np.float32) for name in STL10_CLASSES}
    for split, per_class in [("train", train_per_class), ("test", test_per_class)]:
        for index, name in enumerate(STL10_CLASSES):
            folder = out / split / name
            folder.mkdir(parents=True, exist_ok=True)
            base = colours[name]
            for i in range(per_class):
                array = np.tile(base[None, None, :], (size, size, 1))
                # class-specific stripe frequency so classes are separable
                frequency = 2 + index
                columns = (np.sin(np.arange(size)[None, :] * frequency * np.pi / size) > 0)
                array = array + (columns[..., None] * 45.0)
                array = array + rng.normal(0, 10.0, (size, size, 3))
                Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
                    folder / f"{i:03d}.jpg", quality=90)
    print(f"Fake STL-10 written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fake STL-10 tree for smoke tests.")
    parser.add_argument("--out", default="data/fake_stl10")
    parser.add_argument("--train-per-class", type=int, default=24)
    parser.add_argument("--test-per-class", type=int, default=12)
    args = parser.parse_args()
    make_fake_stl10(args.out, args.train_per_class, args.test_per_class)


if __name__ == "__main__":
    main()
