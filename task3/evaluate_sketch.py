"""Task 3 final evaluation on the unseen Sketch domain.

This file is the ONLY place in Task 3 where target data and labels are loaded.
Run it after all Task 3 training and model-selection decisions are frozen.

    python -m task3.evaluate_sketch --model-run results/t3_dan_dg \
        --erm-run results/t2_source_only
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common.logging import get_logger
from common.metrics import evaluate_classification
from shared.pacs import CLASSES, SOURCE_DOMAINS, find_pacs_root
from shared.pacs_protocol import eval_transform, load_splits, make_loader, subset_from_paths
from task2.evaluation.class_analysis import per_class_rows, top_confusions
from task3.model_loading import load_model_from_run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Task 3 Sketch evaluation (uses target labels).")
    parser.add_argument("--model-run", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--erm-run", default=None,
                        help="Task 2 source-only run dir; enables delta vs ERM")
    return parser.parse_args(argv)


@torch.no_grad()
def collect(model, loader, device):
    logits_all, labels_all, index_all = [], [], []
    for images, labels, _, indices in loader:
        logits, _ = model(images.to(device))
        logits_all.append(logits.cpu().numpy())
        labels_all.append(labels.numpy())
        index_all.append(indices.numpy())
    dataset = loader.dataset
    paths = [dataset.paths()[i] for i in np.concatenate(index_all)]
    return np.concatenate(labels_all), np.concatenate(logits_all), paths


def evaluate_split(model, data_root, splits, domain, split, cfg, device):
    transform = eval_transform(cfg["resize_size"], cfg["crop_size"])
    rel_paths = splits["domains"][domain]["all" if split == "all" else split]
    dataset = subset_from_paths(data_root, rel_paths, transform, domain)
    loader = make_loader(dataset, batch_size=64, shuffle=False, num_workers=0)
    return collect(model, loader, device)


def main(argv=None) -> None:
    args = parse_args(argv)
    logger = get_logger()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg, model, ckpt = load_model_from_run(args.model_run, args.ckpt, device)
    data_root = find_pacs_root(args.data_root or cfg["data_root"])
    splits = load_splits(args.splits or cfg["splits"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.model_run)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_metrics = {}
    for domain in SOURCE_DOMAINS:
        labels, logits, _ = evaluate_split(model, data_root, splits, domain, "val", cfg, device)
        metrics = evaluate_classification(labels, logits.argmax(axis=1), cfg["num_classes"])
        source_metrics[domain] = metrics
        logger.info("source %-13s acc %.4f macro-f1 %.4f", domain, metrics["accuracy"], metrics["macro_f1"])

    labels, logits, paths = evaluate_split(model, data_root, splits, cfg["target_domain"], "all", cfg, device)
    predictions = logits.argmax(axis=1)
    target_metrics = evaluate_classification(labels, predictions, cfg["num_classes"])
    logger.info("target %-13s acc %.4f macro-f1 %.4f", cfg["target_domain"],
                target_metrics["accuracy"], target_metrics["macro_f1"])

    result = {
        "run_id": Path(args.model_run).name,
        "method": cfg["method"],
        "seed": cfg["seed"],
        "checkpoint": str(ckpt),
        "source_validation": source_metrics,
        "source_mean_accuracy": float(np.mean([m["accuracy"] for m in source_metrics.values()])),
        "source_mean_macro_f1": float(np.mean([m["macro_f1"] for m in source_metrics.values()])),
        "source_worst_macro_f1": float(np.min([m["macro_f1"] for m in source_metrics.values()])),
        "target": target_metrics,
        "target_per_class": per_class_rows(labels, predictions, cfg["num_classes"], CLASSES),
        "target_top_confusions": top_confusions(labels, predictions, cfg["num_classes"], CLASSES, k=12),
    }

    if args.erm_run:
        with open(Path(args.erm_run) / "final_metrics.json", "r", encoding="utf-8") as fh:
            erm = json.load(fh)
        result["baseline_run_id"] = erm["run_id"]
        result["target_accuracy_change_vs_erm"] = target_metrics["accuracy"] - erm["target"]["accuracy"]
        result["target_macro_f1_change_vs_erm"] = target_metrics["macro_f1"] - erm["target"]["macro_f1"]

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    np.savez_compressed(out_dir / "target_outputs.npz", labels=labels, logits=logits,
                        paths=np.array(paths))

    with open(out_dir / "target_per_class.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["target_per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(result["target_per_class"])
    with open(out_dir / "target_confusions.csv", "w", encoding="utf-8", newline="") as fh:
        rows = result["target_top_confusions"]
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0].keys()) if rows else
            ["true_class", "predicted_class", "count", "share_of_true_class"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote final_metrics.json to %s", out_dir)


if __name__ == "__main__":
    main()
