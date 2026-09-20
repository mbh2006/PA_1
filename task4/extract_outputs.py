"""Cache frozen model outputs once so every score sees identical examples.

    python -m task4.extract_outputs --run-dir results/t4_vanilla

Produces ``task4/cache/<run_id>.npz`` with logits/features for the CIFAR-10
train split (unaugmented), the CIFAR-10 validation split, the full CIFAR-10
test set, and the two fixed CIFAR-100 unknown groups. Unknown images are read
but never used to influence training or checkpoint selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.logging import get_logger
from task4.data.cifar10 import CifarTest, IndexedCifar, eval_transform, load_split
from task4.data.cifar100_unknowns import Cifar100Unknowns
from task4.model_loading import load_task4_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cache logits/features for OSR scores.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--out", default=None, help="default task4/cache/<run_id>.npz")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                        help="limit images per split (smoke tests only)")
    return parser.parse_args(argv)


@torch.no_grad()
def collect(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    logits_all, features_all, labels_all = [], [], []
    for batch in loader:
        images, labels = batch[0], batch[1]
        logits, features = model(images.to(device))
        logits_all.append(logits.cpu().numpy().astype(np.float32))
        features_all.append(features.cpu().numpy().astype(np.float32))
        labels_all.append(np.asarray(labels))
    return (np.concatenate(logits_all), np.concatenate(features_all),
            np.concatenate(labels_all))


def main(argv=None) -> None:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, ckpt = load_task4_model(args.run_dir, args.ckpt, device)
    root = args.data_root or cfg["data_root"]
    split = load_split(args.split or cfg["split"], root=root)
    transform = eval_transform()

    train_dataset = IndexedCifar(root, True, split["train"], transform)
    val_dataset = IndexedCifar(root, True, split["val"], transform)
    test_dataset = CifarTest(root, transform)
    near_dataset = Cifar100Unknowns(root, "near", transform)
    far_dataset = Cifar100Unknowns(root, "far", transform)

    if args.limit:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(args.limit, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(args.limit, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(args.limit, len(test_dataset))))
        near_dataset = Subset(near_dataset, range(min(args.limit, len(near_dataset))))
        far_dataset = Subset(far_dataset, range(min(args.limit, len(far_dataset))))

    logger.info("extracting outputs with %s", ckpt)
    logits_train, features_train, labels_train = collect(model, train_dataset, device, args.batch_size)
    logits_val, features_val, labels_val = collect(model, val_dataset, device, args.batch_size)
    logits_test, features_test, labels_test = collect(model, test_dataset, device, args.batch_size)
    logits_near, features_near, labels_near = collect(model, near_dataset, device, args.batch_size)
    logits_far, features_far, labels_far = collect(model, far_dataset, device, args.batch_size)

    out_path = Path(args.out) if args.out else Path("task4/cache") / f"{Path(args.run_dir).name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        run_id=np.array(Path(args.run_dir).name),
        method=np.array(cfg["method"]),
        num_known=np.array(cfg["num_classes"]),
        logits_train=logits_train, labels_train=labels_train,
        features_train=features_train,
        logits_val=logits_val, labels_val=labels_val,
        features_val=features_val,
        logits_test=logits_test, labels_test=labels_test,
        features_test=features_test,
        logits_near=logits_near, labels_near=labels_near,
        features_near=features_near,
        logits_far=logits_far, labels_far=labels_far,
        features_far=features_far,
    )
    logger.info("cached %s (train %d, val %d, test %d, near %d, far %d)",
                out_path, len(labels_train), len(labels_val), len(labels_test),
                len(labels_near), len(labels_far))


if __name__ == "__main__":
    main()
