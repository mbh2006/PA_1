"""Final Task 2 evaluation: source-validation metrics + target (Sketch) metrics.

Target labels are read here and ONLY here. Run this after every model,
checkpoint and setting has been frozen. Everything is written to the run's
results folder so the report tables can be assembled from JSON/CSV.

    python -m task2.evaluate_final --run-dir results/task2_dan_seed6304_...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from common.logging import get_logger
from common.metrics import evaluate_classification
from shared.backbone import ResNet18PACS
from shared.pacs import CLASSES, SOURCE_DOMAINS, find_pacs_root
from shared.pacs_protocol import eval_transform, load_splits, make_loader, subset_from_paths
from task2.evaluation.class_analysis import per_class_rows, top_confusions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Final Task 2 evaluation (uses target labels).")
    parser.add_argument("--run-dir", required=True, help="results/<run_id> folder from training")
    parser.add_argument("--ckpt", default=None, help="default: checkpoints/<run_id>/best.pt")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--baseline-run", default=None,
                        help="optional source_only run dir; adds target deltas")
    return parser.parse_args(argv)


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(run_dir: Path, ckpt_path: Path, device: torch.device):
    with open(run_dir / "config.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    model = ResNet18PACS(num_classes=cfg["num_classes"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return cfg, model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (labels, logits, paths) for a loader."""
    model.eval()
    logits_all, labels_all, index_all = [], [], []
    for images, labels, _, indices in loader:
        logits, _ = model(images.to(device))
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
        index_all.append(indices.numpy())
    dataset = loader.dataset
    all_paths = dataset.paths()
    paths = [all_paths[i] for i in np.concatenate(index_all)]
    return np.concatenate(labels_all), np.concatenate(logits_all), paths


def eval_domain(model, data_root, splits, domain: str, split: str, cfg, device):
    transform = eval_transform(cfg["resize_size"], cfg["crop_size"])
    if split == "all":
        rel_paths = splits["domains"][domain]["all"]
    else:
        rel_paths = splits["domains"][domain][split]
    dataset = subset_from_paths(data_root, rel_paths, transform, domain)
    loader = make_loader(dataset, batch_size=64, shuffle=False, num_workers=0)
    labels, logits, paths = collect_predictions(model, loader, device)
    return labels, logits, paths


def main(argv=None) -> None:
    args = parse_args(argv)
    logger = get_logger()
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)

    with open(run_dir / "config.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    data_root = find_pacs_root(args.data_root or cfg["data_root"])
    splits = load_splits(args.splits or cfg["splits"])
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.get("ckpt_root", "checkpoints")) / run_dir.name / "best.pt"
    logger.info("Evaluating %s with %s", run_dir.name, ckpt_path)

    _, model = load_model(run_dir, ckpt_path, device)

    # ------------------------------------------------------------------ source val
    source_metrics: Dict[str, Dict] = {}
    for domain in SOURCE_DOMAINS:
        labels, logits, _ = eval_domain(model, data_root, splits, domain, "val", cfg, device)
        metrics = evaluate_classification(labels, logits.argmax(axis=1), cfg["num_classes"])
        source_metrics[domain] = metrics
        logger.info("source %-13s acc %.4f macro-f1 %.4f", domain, metrics["accuracy"], metrics["macro_f1"])

    # -------------------------------------------------------------------- target
    target_domain = cfg["target_domain"]
    labels, logits, paths = eval_domain(model, data_root, splits, target_domain, "all", cfg, device)
    predictions = logits.argmax(axis=1)
    target_metrics = evaluate_classification(labels, predictions, cfg["num_classes"])
    logger.info("target %-13s acc %.4f macro-f1 %.4f", target_domain,
                target_metrics["accuracy"], target_metrics["macro_f1"])

    rows = per_class_rows(labels, predictions, cfg["num_classes"], CLASSES)
    confusions = top_confusions(labels, predictions, cfg["num_classes"], CLASSES, k=12)

    result = {
        "run_id": run_dir.name,
        "method": cfg["method"],
        "seed": cfg["seed"],
        "checkpoint": str(ckpt_path),
        "source_validation": source_metrics,
        "target": target_metrics,
        "target_per_class": rows,
        "target_top_confusions": confusions,
    }

    # optional delta vs the source-only baseline (target labels already available)
    if args.baseline_run:
        with open(Path(args.baseline_run) / "final_metrics.json", "r", encoding="utf-8") as fh:
            baseline = json.load(fh)
        result["target_accuracy_change_vs_baseline"] = (
            target_metrics["accuracy"] - baseline["target"]["accuracy"]
        )
        result["target_macro_f1_change_vs_baseline"] = (
            target_metrics["macro_f1"] - baseline["target"]["macro_f1"]
        )
        result["baseline_run_id"] = baseline["run_id"]

    with open(run_dir / "final_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    # keep raw logits for later failure analysis / cross-task comparisons
    np.savez_compressed(
        run_dir / "target_outputs.npz",
        labels=labels,
        logits=logits,
        paths=np.array(paths),
    )

    # small CSVs that map directly onto report tables
    import csv
    with open(run_dir / "target_per_class.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(run_dir / "target_confusions.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(confusions[0].keys()) if confusions else
                                ["true_class", "predicted_class", "count", "share_of_true_class"])
        writer.writeheader()
        writer.writerows(confusions)

    logger.info("Wrote final_metrics.json to %s", run_dir)


if __name__ == "__main__":
    main()
