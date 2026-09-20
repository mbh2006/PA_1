"""PACS dataset: 7 object classes in 4 visual domains.

Expected layout (after shared/prepare_pacs.py or a manual download)::

    <root>/
      photo/dog/0001.jpg ...
      art_painting/dog/...
      cartoon/dog/...
      sketch/dog/...

Alternative layouts such as ``<root>/PACS/kfold/photo/...`` are discovered by
``find_pacs_root``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

CLASSES: List[str] = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
DOMAINS: List[str] = ["photo", "art_painting", "cartoon", "sketch"]
SOURCE_DOMAINS: List[str] = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN: str = "sketch"
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_pacs_root(root: str | Path) -> Path:
    """Return the folder that directly contains the four domain folders."""
    root = Path(root)
    direct_candidates = [
        root,
        root / "PACS",
        root / "pacs",
        root / "kfold",
        root / "PACS" / "kfold",
    ]
    for candidate in direct_candidates:
        if candidate.is_dir() and all((candidate / d).is_dir() for d in DOMAINS):
            return candidate
    # generic fallback: search up to a few levels deep
    for candidate in sorted(p for p in root.rglob("*") if p.is_dir()):
        if all((candidate / d).is_dir() for d in DOMAINS):
            return candidate
    raise FileNotFoundError(
        f"Could not find a PACS root under {root!s}. Expected sub-folders {DOMAINS}."
    )


def list_domain_samples(root: str | Path, domain: str) -> List[Tuple[str, int]]:
    """Sorted ``(relative_path, label)`` pairs for one domain.

    Sorting is what makes the seed-6304 split reproducible across machines.
    """
    root = Path(root)
    domain_dir = root / domain
    samples: List[Tuple[str, int]] = []
    for class_name in CLASSES:
        class_dir = domain_dir / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((str(path.relative_to(root)).replace("\\", "/"), CLASS_TO_IDX[class_name]))
    if not samples:
        raise FileNotFoundError(f"No images found for domain '{domain}' under {domain_dir!s}")
    return samples


class PacsDataset(Dataset):
    """Dataset over an explicit list of ``(relative_path, label)`` pairs.

    Returns ``(image, label, domain_index, sample_index)`` so downstream code can
    cache features/logits keyed by dataset position.
    """

    def __init__(
        self,
        root: str | Path,
        samples: Sequence[Tuple[str, int]],
        transform=None,
        domain_index: int = 0,
    ) -> None:
        self.root = Path(root)
        self.samples = list(samples)
        self.transform = transform
        self.domain_index = domain_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.domain_index, index

    # convenience for analysis code
    def paths(self) -> List[str]:
        return [rel for rel, _ in self.samples]

    def labels(self) -> List[int]:
        return [label for _, label in self.samples]
