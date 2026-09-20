"""Datasets and the frozen Task 1 subsets.

Two dataset backends share one split protocol:

* ``stl10``      - torchvision STL-10 (the assignment's recommended dataset);
* ``imagefolder`` - ``<root>/train/<class>/...`` and ``<root>/test/<class>/...``
  used by the local smoke test so the pipeline can run without the 2.6 GB
  download.

The 80/20 stratified split of the official training partition and the
class-balanced 500-image test subset both use seed 6304 and are saved to one
JSON file so that every model sees the same images.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

SUBSET_SEED = 6304
TRAIN_FRAC = 0.8
TEST_SUBSET_SIZE = 500


class IndexedImageDataset(Dataset):
    """Returns ``(PIL image, label, index)`` for a list of underlying indices."""

    def __init__(self, base, indices, common_size: int = 224):
        self.base = base
        self.indices = list(indices)
        self.common_size = common_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        item = self.base[index]
        if isinstance(item[0], (str, Path)):
            image = Image.open(item[0]).convert("RGB")
            label = item[1]
        else:
            image, label = item
        if image.size != (self.common_size, self.common_size):
            image = image.resize((self.common_size, self.common_size), Image.BILINEAR)
        return image, label, index


def _load_base(root: str | Path, dataset: str, split: str):
    if dataset == "stl10":
        from torchvision.datasets import STL10
        base = STL10(root=str(root), split=split, download=True)
        return base, list(base.classes)
    if dataset == "imagefolder":
        folder = Path(root) / split
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        paths = []
        labels = []
        for label, class_name in enumerate(classes):
            for path in sorted((folder / class_name).iterdir()):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    paths.append(path)
                    labels.append(label)
        base = list(zip(paths, labels))
        return base, classes
    raise KeyError(f"unknown dataset backend '{dataset}'")


def build_subsets(root: str | Path, dataset: str = "stl10",
                  out_path: str | Path | None = None) -> dict:
    """Stratified 80/20 train/val split + class-balanced 500 test subset (seed 6304)."""
    train_base, classes = _load_base(root, dataset, "train")
    test_base, test_classes = _load_base(root, dataset, "test")
    assert classes == test_classes, "train/test class names differ"

    labels = np.array([label for _, label in train_base])
    rng = np.random.RandomState(SUBSET_SEED)
    train_indices, val_indices = [], []
    for label in range(len(classes)):
        indices = np.where(labels == label)[0]
        order = rng.permutation(len(indices))
        n_train = int(TRAIN_FRAC * len(indices))
        train_indices.extend(indices[order[:n_train]].tolist())
        val_indices.extend(indices[order[n_train:]].tolist())

    test_labels = np.array([label for _, label in test_base])
    per_class = TEST_SUBSET_SIZE // len(classes)
    test_indices = []
    for label in range(len(classes)):
        indices = np.where(test_labels == label)[0]
        take = min(per_class, len(indices))
        test_indices.extend(rng.permutation(indices)[:take].tolist())
    used_per_class = {classes[label]: int(sum(1 for i in test_indices if test_labels[i] == label))
                      for label in range(len(classes))}

    subsets = {
        "dataset": dataset,
        "seed": SUBSET_SEED,
        "train_frac": TRAIN_FRAC,
        "classes": classes,
        "train": sorted(int(i) for i in train_indices),
        "val": sorted(int(i) for i in val_indices),
        "test_subset": sorted(int(i) for i in test_indices),
        "test_subset_per_class": used_per_class,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(subsets, fh)
    return subsets


def load_subsets(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_dataset(root, dataset: str, split: str, indices, common_size: int = 224):
    base, classes = _load_base(root, dataset, split)
    return IndexedImageDataset(base, indices, common_size), classes
