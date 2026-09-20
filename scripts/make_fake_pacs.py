"""Generate a tiny fake PACS tree for smoke tests (no download, no GPU).

The images are synthetic but class-dependent (each class has a base colour with
noise and a domain tint), so a short smoke run produces non-trivial curves.
Layout is identical to real PACS: <out>/<domain>/<class>/<index>.jpg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from shared.pacs import CLASSES, DOMAINS


def make_fake_pacs(out: str | Path = "data/fake_pacs", per_class: int = 24,
                   size: int = 256, seed: int = 6304) -> Path:
    out = Path(out)
    rng = np.random.RandomState(seed)
    class_colors = {c: rng.randint(40, 215, size=3).astype(np.float32) for c in CLASSES}
    for domain_index, domain in enumerate(DOMAINS):
        tint = np.array([(domain_index * 37) % 80,
                         (domain_index * 53) % 80,
                         (domain_index * 71) % 80], dtype=np.float32)
        for class_name in CLASSES:
            folder = out / domain / class_name
            folder.mkdir(parents=True, exist_ok=True)
            base = class_colors[class_name] + 0.5 * tint
            for index in range(per_class):
                noise = rng.normal(0.0, 18.0, size=(size, size, 3)).astype(np.float32)
                array = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
                Image.fromarray(array).save(folder / f"{index:03d}.jpg", quality=90)
    print(f"Fake PACS written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fake PACS folder for smoke tests.")
    parser.add_argument("--out", default="data/fake_pacs")
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=6304)
    args = parser.parse_args()
    make_fake_pacs(args.out, args.per_class, args.size, args.seed)


if __name__ == "__main__":
    main()
