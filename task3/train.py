"""Task 3 training entry point (DAN-DG / SAM; ERM is reused from Task 2).

Examples::

    python -m task3.train --config task3/configs/dan_dg.yaml
    python -m task3.train --config task3/configs/sam.yaml --rho 0.01   # controlled study

The Task 2 ERM checkpoint is *loaded*, never retrained: the baseline for this
task is exactly the source-only model from Task 2. No Sketch image is touched
anywhere in this file; target evaluation happens only in ``evaluate_sketch.py``.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW

from common.config import config_hash, load_config
from common.logging import RunLogger, get_logger, make_run_id
from common.metrics import evaluate_classification
from common.plotting import plot_history
from common.seed import set_seed
from shared.backbone import ResNet18PACS
from shared.bn_policy import freeze_bn_running_stats
from shared.pacs import SOURCE_DOMAINS, find_pacs_root
from shared.pacs_protocol import (DomainBalancedIterator, build_splits, eval_transform,
                                  load_splits, make_loader, save_splits, split_summary,
                                  subset_from_paths, train_transform)
from task3.methods import build_method


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Task 3 domain-generalisation method.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="results")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    # controlled-study overrides (recorded in the saved config)
    parser.add_argument("--lambda-dg", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    return parser.parse_args(argv)


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(cfg, args):
    data_root = find_pacs_root(args.data_root or cfg["data_root"])
    splits_path = Path(args.splits or cfg["splits"])
    if splits_path.exists():
        splits = load_splits(splits_path)
    else:
        splits = build_splits(data_root, seed=cfg["seed"], train_frac=cfg["train_frac"])
        save_splits(splits, splits_path)
    return data_root, splits


def build_iterator(cfg, data_root, splits, pin_memory):
    transform = train_transform(cfg["resize_size"], cfg["crop_size"])
    loaders = {}
    for domain in SOURCE_DOMAINS:
        dataset = subset_from_paths(data_root, splits["domains"][domain]["train"], transform, domain)
        loaders[domain] = make_loader(
            dataset, batch_size=cfg["batch_size_per_domain"], shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=pin_memory, seed=cfg["seed"],
        )
    return DomainBalancedIterator(loaders, target_loader=None)


def build_val_loaders(cfg, data_root, splits):
    transform = eval_transform(cfg["resize_size"], cfg["crop_size"])
    return {
        domain: make_loader(
            subset_from_paths(data_root, splits["domains"][domain]["val"], transform, domain),
            batch_size=64, shuffle=False, num_workers=cfg["num_workers"],
        )
        for domain in SOURCE_DOMAINS
    }


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate_source_domains(model, val_loaders, num_classes, device):
    model.eval()
    results = {}
    for domain, loader in val_loaders.items():
        all_true, all_pred = [], []
        for images, labels, _, _ in loader:
            logits, _ = model(images.to(device))
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(labels.numpy())
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        results[domain] = evaluate_classification(y_true, y_pred, num_classes)
    return results


def train(cfg: Dict, args, steps_per_epoch_override=None) -> RunLogger:
    logger = get_logger()
    set_seed(cfg["seed"])
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    run = RunLogger(args.run_id or make_run_id("task3", cfg["method"], cfg["seed"]),
                    args.out_root, args.ckpt_root)
    run.save_config(cfg)
    data_root, splits = prepare_data(cfg, args)
    logger.info("run %s | method=%s | device=%s | config_hash=%s",
                run.run_id, cfg["method"], device, config_hash(cfg))
    logger.info("splits: %s", split_summary(splits))

    model = ResNet18PACS(num_classes=cfg["num_classes"]).to(device)
    method = build_method(cfg, device, cfg["num_classes"])
    optimizer = AdamW(model.parameters(), lr=cfg["optimizer"]["lr"],
                      weight_decay=cfg["optimizer"]["weight_decay"])

    iterator = build_iterator(cfg, data_root, splits, pin_memory)
    if steps_per_epoch_override:
        steps_per_epoch = steps_per_epoch_override
    else:
        train_size = sum(len(splits["domains"][d]["train"]) for d in SOURCE_DOMAINS)
        steps_per_epoch = max(1, train_size // (cfg["batch_size_per_domain"] * len(SOURCE_DOMAINS)))
    total_steps = cfg["max_epochs"] * steps_per_epoch
    val_loaders = build_val_loaders(cfg, data_root, splits)

    start_epoch, best_mean_f1, patience = 1, -1.0, 0
    best_row: Dict[str, float] = {}
    if args.resume and (run.ckpt_dir / "last.pt").exists():
        checkpoint = torch.load(run.ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_mean_f1 = checkpoint["best_mean_f1"]
        patience = checkpoint["patience"]
        logger.info("Resumed %s at epoch %d", run.run_id, start_epoch)

    history = []
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        model.train()
        freeze_bn_running_stats(model)
        sums = defaultdict(float)
        started = time.time()
        for step in range(1, steps_per_epoch + 1):
            global_step = (epoch - 1) * steps_per_epoch + step
            progress = (global_step - 1) / max(1, total_steps - 1)
            batch = to_device(iterator.next_batch(), device)
            loss_value, logs = method.step(model, batch, optimizer, progress)
            sums["loss"] += loss_value
            for key, value in logs.items():
                sums[key] += float(value)
        train_stats = {key: value / steps_per_epoch for key, value in sums.items()}

        validation = evaluate_source_domains(model, val_loaders, cfg["num_classes"], device)
        mean_acc = float(np.mean([validation[d]["accuracy"] for d in SOURCE_DOMAINS]))
        mean_f1 = float(np.mean([validation[d]["macro_f1"] for d in SOURCE_DOMAINS]))
        worst_f1 = float(np.min([validation[d]["macro_f1"] for d in SOURCE_DOMAINS]))

        row: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_stats.get("loss", float("nan")),
            "train_cls_loss": train_stats.get("cls_loss", float("nan")),
            "train_mmd": train_stats.get("mmd", float("nan")),
            "val_mean_acc": mean_acc,
            "val_mean_f1": mean_f1,
            "val_worst_f1": worst_f1,
            "epoch_time_s": time.time() - started,
        }
        for domain in SOURCE_DOMAINS:
            row[f"val_{domain}_acc"] = validation[domain]["accuracy"]
            row[f"val_{domain}_f1"] = validation[domain]["macro_f1"]
        history.append(row)
        run.log_row(row)
        logger.info("epoch %d/%d | loss %.4f | mean f1 %.4f worst %.4f (best %.4f, patience %d)",
                    epoch, cfg["max_epochs"], row["train_loss"], mean_f1, worst_f1,
                    max(best_mean_f1, mean_f1), patience)

        improved = mean_f1 > best_mean_f1 + 1e-6
        if improved:
            best_mean_f1 = mean_f1
            patience = 0
            best_row = dict(row)
        else:
            patience += 1

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_mean_f1": best_mean_f1,
            "patience": patience,
            "config": cfg,
        }
        torch.save(checkpoint, run.ckpt_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run.ckpt_dir / "best.pt")
        if patience >= cfg["early_stop_patience"]:
            logger.info("Early stopping at epoch %d", epoch)
            break

    run.save_json("metrics.json", {
        "run_id": run.run_id,
        "method": cfg["method"],
        "seed": cfg["seed"],
        "config_hash": config_hash(cfg),
        "best_mean_source_f1": best_mean_f1,
        "best_row": best_row,
        "epochs_ran": epoch,
        "steps_per_epoch": steps_per_epoch,
        "device": str(device),
        "best_checkpoint": str(run.ckpt_dir / "best.pt"),
    })
    try:
        plot_history(history, ["train_loss", "train_cls_loss", "train_mmd"],
                     run.results_dir / "curves_train.png", title=f"task3 {cfg['method']} - training losses")
        plot_history(history, ["val_mean_f1", "val_worst_f1"],
                     run.results_dir / "curves_val.png", title=f"task3 {cfg['method']} - source validation")
    except Exception as exc:
        logger.warning("Could not write curves: %s", exc)
    logger.info("Finished %s. Best mean source macro-F1: %.4f", run.run_id, best_mean_f1)
    return run


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    cfg["out_root"] = args.out_root
    cfg["ckpt_root"] = args.ckpt_root
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.splits:
        cfg["splits"] = args.splits
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.smoke:
        cfg["max_epochs"] = args.max_epochs or 1
        cfg["early_stop_patience"] = 1
        cfg["num_workers"] = 0
    elif args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    # study overrides are written into the saved config for provenance
    if args.lambda_dg is not None:
        cfg.setdefault("dan_dg", {})["lambda_dg"] = args.lambda_dg
    if args.rho is not None:
        cfg.setdefault("sam", {})["rho"] = args.rho
    train(cfg, args, steps_per_epoch_override=args.steps_per_epoch)


if __name__ == "__main__":
    main()
