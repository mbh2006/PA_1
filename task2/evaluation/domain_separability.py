"""Domain-separability diagnostic for Task 2 (Step 5).

Question: after adaptation, how much *domain* information is still linearly
decodable from the frozen backbone features? Procedure fixed by the assignment:

* freeze the backbone and collect equal numbers of source-validation and target
  features (seed 6304 chooses which images);
* 70/30 split (seed 6304), balanced logistic regression with C = 1;
* held-out accuracy is the separability score; 50% is chance.

A lower score means domain information is harder to recover. It is NOT by itself
evidence that class information survived - always read it together with target
recognition.

    python -m task2.evaluation.domain_separability --run-dir results/task2_dan_seed6304_...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from common.logging import get_logger
from shared.backbone import ResNet18PACS
from shared.pacs import SOURCE_DOMAINS, find_pacs_root
from shared.pacs_protocol import eval_transform, load_splits, make_loader, subset_from_paths

SEED = 6304


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Domain separability of frozen features.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-per-side", type=int, default=None,
                        help="default: min(source val size, target size)")
    return parser.parse_args(argv)


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    for images, targets, _, _ in loader:
        feats = model.features(images.to(device))
        features.append(feats.cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main(argv=None) -> None:
    args = parse_args(argv)
    logger = get_logger()
    run_dir = Path(args.run_dir)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    with open(run_dir / "config.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    data_root = find_pacs_root(args.data_root or cfg["data_root"])
    splits = load_splits(args.splits or cfg["splits"])
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.get("ckpt_root", "checkpoints")) / run_dir.name / "best.pt"

    model = ResNet18PACS(num_classes=cfg["num_classes"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    transform = eval_transform(cfg["resize_size"], cfg["crop_size"])

    # source validation features (all three domains pooled, as specified)
    source_feats = []
    for domain in SOURCE_DOMAINS:
        dataset = subset_from_paths(data_root, splits["domains"][domain]["val"], transform, domain)
        loader = make_loader(dataset, batch_size=64, shuffle=False, num_workers=0)
        feats, _labels = extract_features(model, loader, device)
        source_feats.append(feats)
    source_matrix = np.concatenate(source_feats, axis=0)

    target_domain = cfg["target_domain"]
    target_dataset = subset_from_paths(data_root, splits["domains"][target_domain]["all"], transform, target_domain)
    target_loader = make_loader(target_dataset, batch_size=64, shuffle=False, num_workers=0)
    target_matrix, _ = extract_features(model, target_loader, device)

    n = args.n_per_side or min(len(source_matrix), len(target_matrix))
    rng = np.random.RandomState(SEED)
    source_idx = rng.permutation(len(source_matrix))[:n]
    target_idx = rng.permutation(len(target_matrix))[:n]

    X = np.vstack([source_matrix[source_idx], target_matrix[target_idx]]).astype(np.float64)
    y = np.array([0] * n + [1] * n)

    order = rng.permutation(len(y))
    n_train = int(0.7 * len(y))
    train_idx, test_idx = order[:n_train], order[n_train:]

    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    classifier.fit(X[train_idx], y[train_idx])
    score = float(classifier.score(X[test_idx], y[test_idx]))

    output = {
        "run_id": run_dir.name,
        "method": cfg["method"],
        "score": score,
        "chance": 0.5,
        "n_features": int(X.shape[1]),
        "n_per_domain_class": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "seed": SEED,
        "checkpoint": str(ckpt_path),
    }
    with open(run_dir / "domain_separability.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("%s domain separability: %.4f (chance 0.5)", run_dir.name, score)


if __name__ == "__main__":
    main()
