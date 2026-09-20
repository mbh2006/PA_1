"""Task 3 Step 4b: the common sharpness proxy.

Fixed protocol (identical for ERM, DAN-DG and SAM):

* a fixed validation batch of 32 examples per source domain chosen with seed
  6304, model in evaluation mode;
* one normalised gradient-ascent perturbation with radius 0.05:
  eps = 0.05 * grad L_val / ||grad L_val||_2;
* report Delta_sharp = L_val(theta + eps) - L_val(theta).

This is a standardised local diagnostic, not a proof that one model's whole
loss landscape is flatter.

    python -m task3.evaluation.sharpness --model-run results/t3_sam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.logging import get_logger
from shared.pacs import SOURCE_DOMAINS, find_pacs_root
from shared.pacs_protocol import eval_transform, load_splits, subset_from_paths
from task3.model_loading import load_model_from_run

SEED = 6304
RADIUS = 0.05
PER_DOMAIN = 32


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local sharpness proxy.")
    parser.add_argument("--model-run", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def build_fixed_batch(cfg, data_root, splits, device):
    """32 evaluation examples per source domain, chosen with seed 6304."""
    transform = eval_transform(cfg["resize_size"], cfg["crop_size"])
    rng = np.random.RandomState(SEED)
    images, labels = [], []
    for domain in SOURCE_DOMAINS:
        dataset = subset_from_paths(data_root, splits["domains"][domain]["val"], transform, domain)
        indices = rng.permutation(len(dataset))[:PER_DOMAIN]
        for index in indices:
            image, label, _, _ = dataset[int(index)]
            images.append(image)
            labels.append(label)
    return torch.stack(images).to(device), torch.tensor(labels, device=device)


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

    images, labels = build_fixed_batch(cfg, data_root, splits, device)

    model.eval()
    model.zero_grad(set_to_none=True)
    logits, _ = model(images)
    clean_loss = F.cross_entropy(logits, labels)
    clean_loss.backward()

    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in parameters]), 2)
    epsilon = [RADIUS * p.grad.detach() / (grad_norm + 1e-12) for p in parameters]

    with torch.no_grad():
        for parameter, eps in zip(parameters, epsilon):
            parameter.add_(eps)
        perturbed_loss = F.cross_entropy(model(images)[0], labels)
        for parameter, eps in zip(parameters, epsilon):
            parameter.sub_(eps)

    clean_value = float(clean_loss.detach())
    perturbed_value = float(perturbed_loss)
    output = {
        "model_run": Path(args.model_run).name,
        "checkpoint": str(ckpt),
        "sharpness_delta": perturbed_value - clean_value,
        "loss_clean": clean_value,
        "loss_perturbed": perturbed_value,
        "radius": RADIUS,
        "examples_per_domain": PER_DOMAIN,
        "seed": SEED,
        "grad_norm": float(grad_norm),
    }
    with open(out_dir / "sharpness.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("%s sharpness delta: %.6f (clean %.4f -> perturbed %.4f)",
                args.model_run, output["sharpness_delta"], clean_value, perturbed_value)


if __name__ == "__main__":
    main()
