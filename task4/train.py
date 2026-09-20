"""Task 4 training entry point (Vanilla / GCSC / PROSER).

Examples::

    python -m task4.train --config task4/configs/vanilla.yaml
    python -m task4.train --config task4/configs/gcsc.yaml
    python -m task4.train --config task4/configs/proser.yaml \
        --init-ckpt checkpoints/t4_vanilla/best.pt

Checkpoints are selected on CIFAR-10 validation accuracy only. No unknown
(CIFAR-100) image is loaded anywhere in this file.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from common.config import config_hash, load_config
from common.logging import RunLogger, get_logger, make_run_id
from common.plotting import plot_history
from common.seed import set_seed
from task4.data.cifar10 import (IndexedCifar, build_split, eval_transform, gcsc_transform,
                                load_split, train_transform)
from task4.methods import build_method
from task4.models.resnet_cifar import CifarResNet18


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Task 4 closed-set model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="results")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-ckpt", default=None,
                        help="vanilla checkpoint used to initialise PROSER")
    return parser.parse_args(argv)


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vanilla_into_proser(model, checkpoint_path, logger):
    """Copy a 10-class vanilla checkpoint into a 10+5 PROSER model."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    target = model.state_dict()
    with torch.no_grad():
        for key, value in state.items():
            if key == "net.fc.weight":
                target[key][:10] = value
            elif key == "net.fc.bias":
                target[key][:10] = value
            elif key in target:
                target[key] = value
    model.load_state_dict(target)
    logger.info("Initialised PROSER from %s (first 10 classifier rows + backbone)", checkpoint_path)


def build_loaders(cfg, args, logger):
    root = args.data_root or cfg["data_root"]
    split_path = Path(args.split or cfg["split"])
    if not split_path.exists():
        split = build_split(root, seed=cfg["seed"], out_path=split_path)
    else:
        split = load_split(split_path)
    logger.info("CIFAR-10 split: %d train / %d val (seed %d)",
                len(split["train"]), len(split["val"]), split["seed"])

    train_aug = gcsc_transform() if cfg["method"] == "gcsc" else train_transform()
    train_dataset = IndexedCifar(root, True, split["train"], train_aug)
    val_dataset = IndexedCifar(root, True, split["val"], eval_transform())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=cfg["num_workers"],
    )
    return train_loader, val_loader


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss_sum += float(torch.nn.functional.cross_entropy(model.known_logits(logits), labels)) * labels.size(0)
        correct += int((model.known_logits(logits).argmax(1) == labels).sum())
        total += labels.size(0)
    return correct / max(1, total), loss_sum / max(1, total)


def train(cfg, args, steps_per_epoch_override=None) -> RunLogger:
    logger = get_logger()
    set_seed(cfg["seed"])
    device = resolve_device(args.device)
    run = RunLogger(args.run_id or make_run_id("task4", cfg["method"], cfg["seed"]),
                    args.out_root, args.ckpt_root)
    run.save_config(cfg)
    logger.info("run %s | method=%s | device=%s | config_hash=%s",
                run.run_id, cfg["method"], device, config_hash(cfg))

    train_loader, val_loader = build_loaders(cfg, args, logger)
    steps_per_epoch = steps_per_epoch_override or len(train_loader)

    num_dummies = int(cfg.get("proser", {}).get("num_dummies", 0)) if cfg["method"] == "proser" else 0
    model = CifarResNet18(num_classes=cfg["num_classes"], num_dummies=num_dummies).to(device)
    if cfg["method"] == "proser":
        init_ckpt = args.init_ckpt or cfg.get("init_ckpt")
        if not init_ckpt:
            raise SystemExit("PROSER requires --init-ckpt <vanilla best.pt>")
        load_vanilla_into_proser(model, init_ckpt, logger)

    method = build_method(cfg, device)
    optimizer = SGD(model.parameters(), lr=cfg["optimizer"]["lr"],
                    momentum=cfg["optimizer"]["momentum"],
                    weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])

    start_epoch, best_val_acc, history = 1, -1.0, []
    if args.resume and (run.ckpt_dir / "last.pt").exists():
        checkpoint = torch.load(run.ckpt_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        logger.info("Resumed at epoch %d (best val acc %.4f)", start_epoch, best_val_acc)

    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        model.train()
        started = time.time()
        running_loss = 0.0
        steps = 0
        for step, (images, labels, _) in enumerate(train_loader, start=1):
            if step > steps_per_epoch:
                break
            images, labels = images.to(device), labels.to(device)
            progress = (epoch - 1 + step / steps_per_epoch) / cfg["max_epochs"]
            loss, logs = method.compute_loss(model, images, labels, progress)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            steps += 1
        scheduler.step()
        val_acc, val_loss = validate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, steps),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": time.time() - started,
        }
        history.append(row)
        run.log_row(row)
        logger.info("epoch %d/%d | train loss %.4f | val acc %.4f (best %.4f)",
                    epoch, cfg["max_epochs"], row["train_loss"], val_acc, max(best_val_acc, val_acc))

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": cfg,
        }
        torch.save(checkpoint, run.ckpt_dir / "last.pt")
        if improved:
            torch.save(checkpoint, run.ckpt_dir / "best.pt")

    run.save_json("metrics.json", {
        "run_id": run.run_id,
        "method": cfg["method"],
        "seed": cfg["seed"],
        "config_hash": config_hash(cfg),
        "best_val_accuracy": best_val_acc,
        "epochs_ran": epoch,
        "best_checkpoint": str(run.ckpt_dir / "best.pt"),
    })
    try:
        plot_history(history, ["train_loss", "val_loss"], run.results_dir / "curves_loss.png",
                     title=f"task4 {cfg['method']} - loss")
        plot_history(history, ["val_acc"], run.results_dir / "curves_val.png",
                     title=f"task4 {cfg['method']} - validation accuracy")
    except Exception as exc:
        logger.warning("Could not write curves: %s", exc)
    logger.info("Finished %s. Best val accuracy: %.4f", run.run_id, best_val_acc)
    return run


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    cfg["out_root"] = args.out_root
    cfg["ckpt_root"] = args.ckpt_root
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.split:
        cfg["split"] = args.split
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.smoke:
        cfg["max_epochs"] = args.max_epochs or 1
        cfg["num_workers"] = 0
    elif args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    if args.init_ckpt:
        cfg["init_ckpt"] = args.init_ckpt
    train(cfg, args, steps_per_epoch_override=args.steps_per_epoch)


if __name__ == "__main__":
    main()
