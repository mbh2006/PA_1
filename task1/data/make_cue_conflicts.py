"""Generate cue-conflict images (shape from class A, texture from class B).

For each unordered class pair both directions are generated when feasible:
content=A/style=B and content=B/style=A. A model-free rejection rule is applied
*before* any model prediction is computed:

1. structure preserved - Pearson correlation between the downsampled (64x64)
   grayscale content and stylised images must be >= 0.35;
2. texture shifted - the stylised image's LAB channel means/stds must be closer
   to the style image's than to the content image's.

Accepted and rejected counts are recorded in the manifest; no model prediction
ever influences which images are kept.

    python -m task1.data.make_cue_conflicts --data-root data/stl10 \
        --subsets task1/data/subsets_stl10_seed6304.json --per-direction 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from common.seed import set_seed
from task1.data.dataset import load_subsets, make_dataset
from task1.data.style_transfer import adain_transfer, build_vgg19

DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("airplane", "bird"),
    ("car", "cat"),
    ("deer", "dog"),
    ("horse", "monkey"),
    ("ship", "truck"),
    ("bird", "monkey"),
]
STRUCTURE_THRESHOLD = 0.35


def structure_correlation(content: Image.Image, stylised: Image.Image) -> float:
    a = np.asarray(content.convert("L").resize((64, 64)), dtype=np.float64).ravel()
    b = np.asarray(stylised.convert("L").resize((64, 64)), dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / denominator) if denominator > 0 else 0.0


def lab_statistics(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("LAB"), dtype=np.float64)
    return np.concatenate([array.reshape(-1, 3).mean(axis=0), array.reshape(-1, 3).std(axis=0)])


def style_shifted(content: Image.Image, style: Image.Image, stylised: Image.Image) -> bool:
    content_stats = lab_statistics(content)
    style_stats = lab_statistics(style)
    stylised_stats = lab_statistics(stylised)
    return (np.linalg.norm(stylised_stats - style_stats)
            < np.linalg.norm(stylised_stats - content_stats))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate cue-conflict images.")
    parser.add_argument("--data-root", default="data/stl10")
    parser.add_argument("--dataset", default="stl10")
    parser.add_argument("--subsets", default="task1/data/subsets_stl10_seed6304.json")
    parser.add_argument("--out-dir", default="task1/data/cue_conflicts")
    parser.add_argument("--per-direction", type=int, default=25)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pairs", default=None,
                        help="override with 'a:b,c:d' (class names)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    import torch

    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    subsets = load_subsets(args.subsets)
    classes = subsets["classes"]

    pairs = DEFAULT_PAIRS
    if args.pairs:
        pairs = [tuple(pair.split(":")) for pair in args.pairs.split(",")]

    train_dataset, _ = make_dataset(args.data_root, args.dataset, "train", subsets["train"])
    base = train_dataset.base
    if hasattr(base, "labels"):
        train_labels = np.asarray(base.labels)[train_dataset.indices]
    else:
        train_labels = np.asarray([label for _, label in base])[train_dataset.indices]
    positions: Dict[int, List[int]] = {}
    for position, label in enumerate(train_labels):
        positions.setdefault(int(label), []).append(position)

    vgg = build_vgg19(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {"pairs": {}, "threshold": STRUCTURE_THRESHOLD, "seed": args.seed}
    total_accepted = total_rejected = 0
    rng = np.random.RandomState(args.seed)

    for a_name, b_name in pairs:
        a_index, b_index = classes.index(a_name), classes.index(b_name)
        for content_name, style_name, content_index, style_index in [
            (a_name, b_name, a_index, b_index),
            (b_name, a_name, b_index, a_index),
        ]:
            key = f"{content_name}_content_{style_name}_style"
            content_pool = positions[content_index]
            style_pool = positions[style_index]
            n = min(args.per_direction, len(content_pool), len(style_pool))
            content_positions = rng.permutation(len(content_pool))[:n]
            style_positions = rng.permutation(len(style_pool))[:n]
            accepted = rejected = 0
            for i in range(n):
                content_image, _, content_original_index = train_dataset[int(content_pool[content_positions[i]])]
                style_image, _, style_original_index = train_dataset[int(style_pool[style_positions[i]])]
                stylised = adain_transfer(content_image, style_image, vgg, device,
                                          steps=args.steps, seed=args.seed + i)
                correlation = structure_correlation(content_image, stylised)
                if correlation >= STRUCTURE_THRESHOLD and style_shifted(content_image, style_image, stylised):
                    filename = f"{key}_{accepted:03d}.png"
                    stylised.save(out_dir / filename)
                    accepted += 1
                else:
                    rejected += 1
            manifest["pairs"][key] = {
                "content_class": content_name,
                "style_class": style_name,
                "candidates": n,
                "accepted": accepted,
                "rejected": rejected,
            }
            total_accepted += accepted
            total_rejected += rejected
            print(f"{key}: accepted {accepted} / {n}", flush=True)

    manifest["total_accepted"] = total_accepted
    manifest["total_rejected"] = total_rejected
    manifest["rejection_rule"] = (
        f"structure correlation >= {STRUCTURE_THRESHOLD} AND LAB statistics closer to style than content")
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"total accepted {total_accepted}, rejected {total_rejected}; manifest in {out_dir}")


if __name__ == "__main__":
    main()
