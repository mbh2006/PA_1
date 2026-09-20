"""Task 2 training entry point - one pipeline for every method.

Examples::

    # real run (Kaggle GPU)
    python -m task2.train --config task2/configs/dann.yaml

    # resume after a Kaggle session ran out of time
    python -m task2.train --config task2/configs/dann.yaml --resume \
        --run-id task2_dann_seed6304_20260920-120000

    # local CPU smoke test on fake data (no PACS download needed)
    python -m task2.train --config task2/configs/source_only.yaml --smoke \
        --data-root data/fake_pacs --splits data/fake_pacs/splits_seed6304.json \
        --out-root results_smoke --ckpt-root checkpoints_smoke --run-id smoke_source_only

What is fixed across methods (per the assignment): backbone + head, full
fine-tuning, optimizer (AdamW 1e-4 / wd 1e-4), augmentation, domain-balanced
batching, epoch budget 30, early stopping 5 on mean source-validation macro-F1,
seed 6304, and the frozen-BN running-statistics policy.
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
from task2.methods import build_method


# ---------------------------------------------------------------------------- helpers
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Task 2 adaptation method on PACS.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="results")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    parser.add_argument("--smoke", action="store_true", help="tiny run for pipeline testing")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(cfg: Dict, args) -> tuple:
    """Find the dataset root and load (or build once) the frozen split file."""
    data_root = find_pacs_root(args.data_root or cfg["data_root"])
    splits_path = Path(args.splits or cfg["splits"])
    if splits_path.exists():
        splits = load_splits(splits_path)
    else:
        splits = build_splits(data_root, seed=cfg["seed"], train_frac=cfg["train_frac"])
        save_splits(splits, splits_path)
        get_logger().info("Created new split file at %s", splits_path)
    return data_root, splits


def build_iterator(cfg: Dict, data_root, splits: Dict, method, pin_memory: bool) -> DomainBalancedIterator:
    transform = train_transform(cfg["resize_size"], cfg["crop_size"])
    source_loaders = {}
    for domain in SOURCE_DOMAINS:
        dataset = subset_from_paths(data_root, splits["domains"][domain]["train"], transform, domain)
        source_loaders[domain] = make_loader(
            dataset,
            batch_size=cfg["batch_size_per_domain"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=pin_memory,
            seed=cfg["seed"],
        )
    target_loader = None
    if method.requires_target:
        target_domain = cfg["target_domain"]
        dataset = subset_from_paths(data_root, splits["domains"][target_domain]["all"], transform, target_domain)
        target_loader = make_loader(
            dataset,
            batch_size=cfg["target_batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            drop_last=True,
            pin_memory=pin_memory,
            seed=cfg["seed"],
        )
    return DomainBalancedIterator(source_loaders, target_loader)


def build_val_loaders(cfg: Dict, data_root, splits: Dict):
    transform = eval_transform(cfg["resize_size"], cfg["crop_size"])
    loaders = {}
    for domain in SOURCE_DOMAINS:
        dataset = subset_from_paths(data_root, splits["domains"][domain]["val"], transform, domain)
        loaders[domain] = make_loader(
            dataset, batch_size=64, shuffle=False, num_workers=cfg["num_workers"]
        )
    return loaders


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate_source_domains(model, val_loaders, num_classes: int, device) -> Dict[str, Dict]:
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


# ------------------------------------------------------------------------------- train
def train(cfg: Dict, args, steps_per_epoch_override: int | None = None) -> RunLogger:
    logger = get_logger()
    set_seed(cfg["seed"])
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"

    run = RunLogger(args.run_id or make_run_id("task2", cfg["method"], cfg["seed"]),
                    args.out_root, args.ckpt_root)
    run.save_config(cfg)
    data_root, splits = prepare_data(cfg, args)
    logger.info("run %s | method=%s | device=%s | config_hash=%s",
                run.run_id, cfg["method"], device, config_hash(cfg))
    logger.info("splits: %s", split_summary(splits))

    model = ResNet18PACS(num_classes=cfg["num_classes"]).to(device)
    method = build_method(cfg, device, model.feature_dim, cfg["num_classes"])
    parameters = list(model.parameters()) + list(method.extra_parameters())
    optimizer = AdamW(parameters, lr=cfg["optimizer"]["lr"],
                      weight_decay=cfg["optimizer"]["weight_decay"])

    iterator = build_iterator(cfg, data_root, splits, method, pin_memory)
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
        if checkpoint.get("method"):
            method.load_state_dict(checkpoint["method"])
        start_epoch = checkpoint["epoch"] + 1
        best_mean_f1 = checkpoint["best_mean_f1"]
        patience = checkpoint["patience"]
        logger.info("Resumed %s at epoch %d (best mean f1 %.4f)", run.run_id, start_epoch, best_mean_f1)

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
            loss, logs = method.compute(model, batch, progress)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums["loss"] += float(loss.detach())
            for key, value in logs.items():
                sums[key] += float(value)
        train_stats = {key: value / steps_per_epoch for key, value in sums.items()}

        validation = evaluate_source_domains(model, val_loaders, cfg["num_classes"], device)
        mean_acc = float(np.mean([validation[d]["accuracy"] for d in SOURCE_DOMAINS]))
        mean_f1 = float(np.mean([validation[d]["macro_f1"] for d in SOURCE_DOMAINS]))

        row: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_stats.get("loss", float("nan")),
            "train_cls_loss": train_stats.get("cls_loss", float("nan")),
            "train_mmd": train_stats.get("mmd", float("nan")),
            "train_dom_loss": train_stats.get("dom_loss", float("nan")),
            "train_dom_acc": train_stats.get("dom_acc", float("nan")),
            "train_alpha": train_stats.get("alpha", float("nan")),
            "val_mean_acc": mean_acc,
            "val_mean_f1": mean_f1,
            "epoch_time_s": time.time() - started,
        }
        for domain in SOURCE_DOMAINS:
            row[f"val_{domain}_acc"] = validation[domain]["accuracy"]
            row[f"val_{domain}_f1"] = validation[domain]["macro_f1"]
        history.append(row)
        run.log_row(row)
        logger.info(
            "epoch %d/%d | loss %.4f | mean source f1 %.4f (best %.4f, patience %d)",
            epoch, cfg["max_epochs"], row["train_loss"], mean_f1, max(best_mean_f1, mean_f1), patience,
        )

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
            "method": method.extra_state_dict(),
            "epoch": epoch,
            "best_mean_f1": best_mean_f1,
            "patience": patience,
            "config": cfg,
        }
        torch.save(checkpoint, run.ckpt_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run.ckpt_dir / "best.pt")

        if patience >= cfg["early_stop_patience"]:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, cfg["early_stop_patience"])
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

    # training curves (required evidence: classification / alignment-loss curves)
    try:
        plot_history(history, ["train_loss", "train_cls_loss", "train_mmd", "train_dom_loss"],
                     run.results_dir / "curves_train.png", title=f"task2 {cfg['method']} - training losses")
        plot_history(history, ["val_mean_f1", "val_mean_acc"],
                     run.results_dir / "curves_val.png", title=f"task2 {cfg['method']} - source validation")
    except Exception as exc:  # plotting must never kill a finished run
        logger.warning("Could not write training curves: %s", exc)

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
    train(cfg, args, steps_per_epoch_override=args.steps_per_epoch)


if __name__ == "__main__":
    main()
