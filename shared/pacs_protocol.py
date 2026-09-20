"""The shared PACS protocol for Tasks 2 and 3.

Everything that must be *identical* between the two tasks lives here:

* the stratified 80/20 source split (seed 6304), saved to a JSON file so it can
  be inspected, committed and reused instead of being regenerated;
* preprocessing / augmentation (Resize 256, random 224 crop + hflip; center crop
  for evaluation; ImageNet normalization);
* domain-balanced batch iteration: 8 examples from each source domain per update
  (plus 24 target examples in Task 2).

The split JSON is the contract between Tasks 2 and 3. Do not regenerate it with
a different seed or file order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from common.seed import SEED, generator, seed_worker
from shared.pacs import (CLASSES, DOMAINS, SOURCE_DOMAINS, TARGET_DOMAIN, PacsDataset,
                         list_domain_samples)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- transforms
def train_transform(resize_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    """Resize(256) -> RandomCrop(224) -> HFlip -> ToTensor -> ImageNet normalize."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(resize_size: int = 256, crop_size: int = 224) -> transforms.Compose:
    """Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet normalize."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ------------------------------------------------------------------------------ splits
def build_splits(root: str | Path, seed: int = SEED, train_frac: float = 0.8) -> Dict:
    """Stratified per-domain 80/20 split of every source domain (seed 6304).

    Returns a dict::

        {"seed": 6304, "train_frac": 0.8,
         "domains": {"photo": {"train": [...], "val": [...]},
                     ..., "sketch": {"all": [...]}}}
    """
    root = Path(root)
    rng = np.random.RandomState(seed)
    out: Dict = {"seed": seed, "train_frac": train_frac, "domains": {}}

    for domain in SOURCE_DOMAINS:
        samples = list_domain_samples(root, domain)
        by_class: Dict[int, List[str]] = {c: [] for c in range(len(CLASSES))}
        for rel_path, label in samples:
            by_class[label].append(rel_path)

        train_paths: List[str] = []
        val_paths: List[str] = []
        for label in range(len(CLASSES)):
            paths = by_class[label]
            order = rng.permutation(len(paths))
            n_train = int(train_frac * len(paths))
            train_idx = order[:n_train]
            val_idx = order[n_train:]
            train_paths.extend(paths[i] for i in train_idx)
            val_paths.extend(paths[i] for i in val_idx)

        out["domains"][domain] = {
            "train": sorted(train_paths),
            "val": sorted(val_paths),
        }

    # the complete target domain (labels are never needed for this file)
    out["domains"][TARGET_DOMAIN] = {
        "all": [rel for rel, _ in list_domain_samples(root, TARGET_DOMAIN)]
    }
    return out


def save_splits(splits: Dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=1)
    return path


def load_splits(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_summary(splits: Dict) -> Dict[str, Dict[str, int]]:
    return {
        domain: {split: len(paths) for split, paths in splits["domains"][domain].items()}
        for domain in splits["domains"]
    }


# ---------------------------------------------------------------------------- datasets
def subset_from_paths(root: str | Path, rel_paths: Sequence[str], transform, domain: str) -> PacsDataset:
    """Build a PacsDataset from relative paths (labels read from folder names)."""
    from shared.pacs import CLASS_TO_IDX

    domain_index = DOMAINS.index(domain)
    samples: List[Tuple[str, int]] = []
    for rel in rel_paths:
        class_name = Path(rel).parent.name
        samples.append((rel, CLASS_TO_IDX[class_name]))
    return PacsDataset(root, samples, transform=transform, domain_index=domain_index)


# ----------------------------------------------------------------------------- loaders
def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    seed: int = SEED,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator(seed) if shuffle else None,
        persistent_workers=False,
    )


class DomainBalancedIterator:
    """Infinite iterator producing one training update at a time.

    Each update contains exactly ``batch_size_per_domain`` examples from every
    source domain (8 by default) and, when a target loader is given, the target
    block (24 examples). Loaders are cycled independently, so domains with
    different dataset sizes stay balanced.
    """

    def __init__(self, source_loaders: Dict[str, DataLoader], target_loader: Optional[DataLoader] = None):
        self.source_loaders = dict(source_loaders)
        self.target_loader = target_loader
        self._iters: Dict[str, object] = {}
        self._target_iter = None

    def _next_from(self, name: str, loader: DataLoader):
        if name not in self._iters:
            self._iters[name] = iter(loader)
        iterator = self._iters[name]
        try:
            return next(iterator)
        except StopIteration:
            self._iters[name] = iter(loader)
            return next(self._iters[name])

    def next_batch(self) -> Dict[str, torch.Tensor]:
        xs, ys, domains = [], [], []
        for name, loader in self.source_loaders.items():
            x, y, d, _ = self._next_from(name, loader)
            xs.append(x)
            ys.append(y)
            domains.append(d)
        batch = {
            "source_x": torch.cat(xs, dim=0),
            "source_y": torch.cat(ys, dim=0),
            "source_domain": torch.cat(domains, dim=0),
        }
        if self.target_loader is not None:
            if self._target_iter is None:
                self._target_iter = iter(self.target_loader)
            try:
                target_x, _, _, _ = next(self._target_iter)
            except StopIteration:
                self._target_iter = iter(self.target_loader)
                target_x, _, _, _ = next(self._target_iter)
            batch["target_x"] = target_x
        return batch
