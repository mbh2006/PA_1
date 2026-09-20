"""Task 3 Step 4a: source-domain separability.

Freeze the backbone, collect balanced features from the three source
validation sets, train a multinomial logistic regression (C = 1) on a 70/30
split (seed 6304) to predict Photo / Art Painting / Cartoon, and report
held-out accuracy. Chance is 33.3%.

A lower score means the source domains are harder to tell apart in feature
space. It does NOT by itself establish that class information or unseen-domain
performance improved.

    python -m task3.evaluation.source_domain_separability --model-run results/t3_dan_dg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from common.logging import get_logger
from shared.pacs import SOURCE_DOMAINS, find_pacs_root
from shared.pacs_protocol import eval_transform, load_splits, make_loader, subset_from_paths
from task3.model_loading import load_model_from_run

SEED = 6304


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Three-way source-domain separability.")
    parser.add_argument("--model-run", required=True, help="results/<run_id> folder")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--out-dir", default=None, help="default: the model run folder")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


@torch.no_grad()
def extract_features(model, loader, device):
    features = []
    for images, _, _, _ in loader:
        features.append(model.features(images.to(device)).cpu().numpy())
    return np.concatenate(features, axis=0)


def main(argv=None) -> None:
    args = parse_args(argv)
    logger = get_logger()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.model_run)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, ckpt = load_model_from_run(args.model_run, args.ckpt, device)
    data_root = find_pacs_root(args.data_root or cfg["data_root"])
    splits = load_splits(args.splits or cfg["splits"])
    transform = eval_transform(cfg["resize_size"], cfg["crop_size"])

    per_domain_features = []
    for domain in SOURCE_DOMAINS:
        dataset = subset_from_paths(data_root, splits["domains"][domain]["val"], transform, domain)
        loader = make_loader(dataset, batch_size=64, shuffle=False, num_workers=0)
        per_domain_features.append(extract_features(model, loader, device))

    n = min(len(features) for features in per_domain_features)
    rng = np.random.RandomState(SEED)
    X = np.vstack([features[rng.permutation(len(features))[:n]] for features in per_domain_features])
    y = np.concatenate([np.full(n, index) for index in range(len(SOURCE_DOMAINS))])

    order = rng.permutation(len(y))
    n_train = int(0.7 * len(y))
    train_idx, test_idx = order[:n_train], order[n_train:]

    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    classifier.fit(X[train_idx], y[train_idx])
    score = float(classifier.score(X[test_idx], y[test_idx]))

    output = {
        "model_run": Path(args.model_run).name,
        "checkpoint": str(ckpt),
        "score": score,
        "chance": 1.0 / len(SOURCE_DOMAINS),
        "n_per_domain": int(n),
        "n_features": int(X.shape[1]),
        "seed": SEED,
    }
    with open(out_dir / "source_domain_separability.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("%s source-domain separability: %.4f (chance %.3f)",
                args.model_run, score, 1.0 / len(SOURCE_DOMAINS))


if __name__ == "__main__":
    main()
