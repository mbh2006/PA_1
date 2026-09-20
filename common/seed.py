"""Deterministic seeding helpers.

The assignment fixes seed 6304 for every comparison, so every script calls
``set_seed()`` (or ``set_seed(6304)``) before building models or samplers.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

SEED = 6304


def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA).

    ``deterministic=True`` additionally makes cuDNN deterministic; it is slower
    and only needed if you want bit-exact reruns. The default keeps PyTorch's
    fastest kernels while the *data order and weight init* remain seeded.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - older torch
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoaders so workers are seeded deterministically."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generator(seed: int = SEED) -> torch.Generator:
    """A torch.Generator seeded once; pass it to DataLoader(shuffle=True)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
