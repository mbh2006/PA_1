"""CIFAR-100 unknown groups used ONLY at evaluation time.

The two groups are fixed by the assignment and must not be revised after
results are seen. Each group has 800 images (8 fine classes x 100). CIFAR-100
*train* images are never used anywhere in Task 4.
"""
from __future__ import annotations

from typing import Dict, List

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

NEAR_UNKNOWN: List[str] = [
    "bus", "pickup_truck", "motorcycle", "tractor",
    "wolf", "fox", "leopard", "camel",
]
FAR_UNKNOWN: List[str] = [
    "bottle", "bowl", "chair", "clock",
    "keyboard", "mushroom", "sunflower", "wardrobe",
]
GROUPS: Dict[str, List[str]] = {"near": NEAR_UNKNOWN, "far": FAR_UNKNOWN}


class Cifar100Unknowns(Dataset):
    """CIFAR-100 test images restricted to one unknown group.

    Returns ``(image, fine_label_index, group_index)`` where fine_label_index is
    the CIFAR-100 fine label (0-99) and group_index is 0 for near, 1 for far.
    """

    def __init__(self, root, group: str, transform):
        if group not in GROUPS:
            raise KeyError(f"unknown group must be one of {sorted(GROUPS)}")
        self.group = group
        self.base = CIFAR100(root=str(root), train=False, download=True)
        fine_names = self.base.classes
        wanted = GROUPS[group]
        self.fine_labels = [fine_names.index(name) for name in wanted]
        label_set = set(self.fine_labels)
        self.indices = [i for i, label in enumerate(self.base.targets) if label in label_set]
        self.transform = transform
        self.group_index = 0 if group == "near" else 1
        self.label_names = {label: fine_names[label] for label in self.fine_labels}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image, label = self.base[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.group_index
