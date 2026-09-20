"""CIFAR-10 data and the fixed stratified 90/10 split (seed 6304).

The official training partition is split once into train/validation; the
validation part is used for checkpoint selection AND for calibrating the
rejection thresholds (known data only). The official test partition is used for
final known-class evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
DEFAULT_SPLIT = Path("task4/data/splits/cifar10_seed6304.json")


def train_transform():
    """Random crop (pad 4) + horizontal flip + normalise."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def gcsc_transform():
    """The GCSC recipe: identical to vanilla plus RandAugment(2, 9) inserted
    after crop/flip and before conversion and normalisation."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def build_split(root: str | Path, seed: int = 6304, val_frac: float = 0.1,
                out_path: str | Path | None = DEFAULT_SPLIT) -> dict:
    """Stratified 90/10 split of the official CIFAR-10 train partition."""
    dataset = CIFAR10(root=str(root), train=True, download=True)
    targets = np.array(dataset.targets)
    rng = np.random.RandomState(seed)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = np.where(targets == label)[0]
        order = rng.permutation(len(indices))
        n_val = int(val_frac * len(indices))
        val_indices.extend(indices[order[:n_val]].tolist())
        train_indices.extend(indices[order[n_val:]].tolist())
    split = {
        "seed": seed,
        "val_frac": val_frac,
        "train": sorted(int(i) for i in train_indices),
        "val": sorted(int(i) for i in val_indices),
    }
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(split, fh)
    return split


def load_split(path: str | Path = DEFAULT_SPLIT, root: str | Path = "data") -> dict:
    path = Path(path)
    if not path.exists():
        return build_split(root, out_path=path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class IndexedCifar(Dataset):
    """CIFAR-10 restricted to a list of indices; returns (image, label, index)."""

    def __init__(self, root, train: bool, indices, transform):
        self.base = CIFAR10(root=str(root), train=train, download=True)
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, label = self.base[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, index


class CifarTest(Dataset):
    """Full official test set; returns (image, label, index)."""

    def __init__(self, root, transform):
        self.base = CIFAR10(root=str(root), train=False, download=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, index
