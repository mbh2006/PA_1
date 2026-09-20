"""Task 1 pipeline in cached stages.

    python -m task1.run_task1 --config task1/configs/base.yaml --stage all

Stages:

* ``prepare``   - write the frozen subsets JSON (stratified 80/20 split of the
  official training partition + class-balanced 500-image test subset, seed 6304);
* ``conflicts`` - generate the AdaIN cue-conflict images with the model-free
  rejection rule;
* ``cache``     - extract frozen features for every condition and backbone
  (clean / grayscale / hue / patch shuffle / translations / conflicts);
* ``heads``     - train the linear classifier head per backbone on clean train
  features (AdamW 1e-3, wd 1e-4, <=50 epochs, early stop 5, seed 6304);
* ``analysis``  - accuracy/F1/confidence, prediction consistency, shape bias and
  coverage, representation stability, translation curves and t-SNE figures.

Smoke mode: ``--limit`` caps the number of eval images and ``--backbones``
selects a subset; the same pipeline is used on Kaggle with the full data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common.config import config_hash, load_config
from common.logging import get_logger, make_run_id
from common.seed import set_seed
from task1.analysis.metrics import (classification_metrics, prediction_consistency,
                                    shape_bias_coverage, shape_texture_counts, softmax)
from task1.analysis.representation import cosine_stability, tsne_figure
from task1.data.dataset import build_subsets, load_subsets, make_dataset
from task1.data.interventions import (INTERVENTIONS, TRANSLATION_DIRECTIONS,
                                      TRANSLATION_DISPLACEMENTS, translate)
from task1.models.backbones import build_backbone


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Task 1 pipeline.")
    parser.add_argument("--config", default="task1/configs/base.yaml")
    parser.add_argument("--stage", default="all",
                        choices=["prepare", "conflicts", "cache", "heads", "analysis", "all"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=None, help="override cfg['data_root']")
    parser.add_argument("--limit", type=int, default=None, help="cap eval images (smoke)")
    parser.add_argument("--backbones", default=None, help="comma-separated subset (smoke)")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def results_dir(cfg) -> Path:
    path = Path(cfg["results_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- stages
def stage_prepare(cfg, args) -> dict:
    path = Path(cfg["subsets"])
    if path.exists():
        subsets = load_subsets(path)
    else:
        subsets = build_subsets(cfg["data_root"], cfg["dataset"], out_path=path)
    get_logger().info("subsets: %d train / %d val / %d test-subset images",
                      len(subsets["train"]), len(subsets["val"]), len(subsets["test_subset"]))
    return subsets


def stage_conflicts(cfg, args) -> None:
    from task1.data import make_cue_conflicts as conflicts
    pairs = ",".join(":".join(pair) for pair in cfg["conflict_pairs"])
    argv = ["--data-root", cfg["data_root"], "--dataset", cfg["dataset"],
            "--subsets", cfg["subsets"], "--out-dir", cfg["conflicts_dir"],
            "--per-direction", str(cfg["conflict_per_direction"]),
            "--steps", str(cfg["conflict_steps"]), "--pairs", pairs]
    if args.device:
        argv += ["--device", args.device]
    conflicts.main(argv)


def _condition_list(cfg):
    conditions = ["clean", "grayscale", "hue", "shuffle"]
    for pixels in TRANSLATION_DISPLACEMENTS:
        if pixels == 0:
            continue
        for direction in TRANSLATION_DIRECTIONS:
            conditions.append(f"translate_{direction}_{pixels}")
    return conditions


def _apply_condition(image, condition, index):
    if condition in INTERVENTIONS:
        return INTERVENTIONS[condition](image, index)
    if condition.startswith("translate_"):
        _, direction, pixels = condition.split("_")
        return translate(image, int(pixels), direction)
    raise KeyError(condition)


@torch.no_grad()
def _extract(backbone, dataset, condition, device, batch_size):
    features, labels, indices = [], [], []
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        images = [_apply_condition(image, condition, int(index)) for image, _, index in batch]
        tensor = backbone.pil_to_tensor(images, device)
        features.append(backbone.features(tensor).cpu().numpy().astype(np.float32))
        labels.extend(int(label) for _, label, _ in batch)
        indices.extend(int(index) for _, _, index in batch)
    return np.concatenate(features), np.asarray(labels), np.asarray(indices)


def stage_cache(cfg, args, subsets) -> None:
    device = resolve_device(args.device)
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone_names = (args.backbones.split(",") if args.backbones else cfg["backbones"])
    limit = args.limit

    test_indices = subsets["test_subset"][:limit] if limit else subsets["test_subset"]
    test_dataset, _ = make_dataset(cfg["data_root"], cfg["dataset"], "test", test_indices,
                                   cfg["common_size"])
    train_indices = subsets["train"][:limit] if limit else subsets["train"]
    val_indices = subsets["val"][:limit] if limit else subsets["val"]

    for name in backbone_names:
        backbone = build_backbone(name, device)
        log = get_logger()
        log.info("[%s] caching train/val features", name)
        for split, split_indices in [("train", train_indices), ("val", val_indices)]:
            dataset, _ = make_dataset(cfg["data_root"], cfg["dataset"], "train", split_indices,
                                      cfg["common_size"])
            features, labels, indices = _extract(backbone, dataset, "clean", device, args.batch_size)
            np.savez_compressed(cache_dir / f"{name}_{split}.npz",
                                features=features, labels=labels, indices=indices)

        for condition in _condition_list(cfg):
            features, labels, indices = _extract(backbone, test_dataset, condition, device,
                                                 args.batch_size)
            np.savez_compressed(cache_dir / f"{name}_{condition}.npz",
                                features=features, labels=labels, indices=indices)
            log.info("[%s] cached condition %s", name, condition)

        if getattr(backbone, "supports_zero_shot", False):
            text = backbone.text_features(subsets["classes"]).cpu().numpy()
            scale = float(backbone.model.logit_scale.exp())
            np.savez(cache_dir / "clip_zero_shot.npz", text_features=text, scale=scale)

        conflict_dir = Path(cfg["conflicts_dir"])
        manifest_path = conflict_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            classes = subsets["classes"]
            files, content_labels, style_labels, features = [], [], [], []
            for key, info in manifest["pairs"].items():
                for path in sorted(conflict_dir.glob(f"{key}_*.png")):
                    from PIL import Image
                    image = Image.open(path).convert("RGB").resize(
                        (cfg["common_size"], cfg["common_size"]), Image.BILINEAR)
                    tensor = backbone.pil_to_tensor([image], device)
                    features.append(backbone.features(tensor).cpu().numpy().astype(np.float32)[0])
                    files.append(path.name)
                    content_labels.append(classes.index(info["content_class"]))
                    style_labels.append(classes.index(info["style_class"]))
            if not features:
                log.warning("[%s] no accepted cue-conflict images; skipping conflict cache", name)
            else:
                np.savez_compressed(cache_dir / f"{name}_conflict.npz",
                                    features=np.stack(features),
                                    content_labels=np.asarray(content_labels),
                                    style_labels=np.asarray(style_labels), files=np.array(files))
                log.info("[%s] cached %d conflict images", name, len(files))
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()


def stage_heads(cfg, args, subsets) -> None:
    device = resolve_device(args.device)
    set_seed(cfg["seed"])
    cache_dir = Path(cfg["cache_dir"])
    heads_dir = results_dir(cfg) / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    backbone_names = (args.backbones.split(",") if args.backbones else cfg["backbones"])
    log = get_logger()

    for name in backbone_names:
        train = np.load(cache_dir / f"{name}_train.npz")
        val = np.load(cache_dir / f"{name}_val.npz")
        x_train = torch.from_numpy(train["features"]).to(device)
        y_train = torch.from_numpy(train["labels"]).to(device)
        x_val = torch.from_numpy(val["features"]).to(device)
        y_val = torch.from_numpy(val["labels"]).to(device)

        head = torch.nn.Linear(x_train.shape[1], len(subsets["classes"])).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["head"]["lr"],
                                      weight_decay=cfg["head"]["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()
        best_accuracy, patience, best_state, history = -1.0, 0, None, []

        for epoch in range(1, cfg["head"]["max_epochs"] + 1):
            head.train()
            permutation = torch.randperm(len(x_train), device=device)
            epoch_loss = 0.0
            for start in range(0, len(x_train), cfg["head"]["batch_size"]):
                batch = permutation[start:start + cfg["head"]["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(head(x_train[batch]), y_train[batch])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch) / len(x_train)
            head.eval()
            with torch.no_grad():
                val_accuracy = float((head(x_val).argmax(1) == y_val).float().mean())
            history.append({"epoch": epoch, "train_loss": epoch_loss, "val_accuracy": val_accuracy})
            log.info("[%s head] epoch %d val acc %.4f", name, epoch, val_accuracy)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg["head"]["patience"]:
                    break

        torch.save({"state_dict": best_state, "backbone": name, "best_val_accuracy": best_accuracy,
                    "config": cfg["head"]}, heads_dir / f"{name}.pt")
        with open(heads_dir / f"{name}_history.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_accuracy"])
            writer.writeheader()
            writer.writerows(history)
        log.info("[%s] best head val accuracy %.4f", name, best_accuracy)


def _head_logits(cfg, name, features, device):
    head_path = results_dir(cfg) / "heads" / f"{name}.pt"
    checkpoint = torch.load(head_path, map_location=device, weights_only=False)
    weight = checkpoint["state_dict"]["weight"]
    bias = checkpoint["state_dict"]["bias"]
    with torch.no_grad():
        return (torch.from_numpy(features).to(device) @ weight.t() + bias).cpu().numpy()


def stage_analysis(cfg, args, subsets) -> None:
    device = resolve_device(args.device)
    cache_dir = Path(cfg["cache_dir"])
    out_dir = results_dir(cfg)
    log = get_logger()
    num_classes = len(subsets["classes"])
    limit = args.limit
    backbone_names = (args.backbones.split(",") if args.backbones else cfg["backbones"])

    clip_data = None
    clip_path = cache_dir / "clip_zero_shot.npz"
    if clip_path.exists():
        clip_data = np.load(clip_path)

    report = {"subsets": {"train": len(subsets["train"]), "val": len(subsets["val"]),
                          "test_subset": len(subsets["test_subset"])},
              "conditions": {}, "conflicts": {}, "translation": {}, "stability": {},
              "backbones": {}, "config_hash": config_hash(cfg)}

    for name in backbone_names:
        clean = np.load(cache_dir / f"{name}_clean.npz")
        clean_features = clean["features"][:limit] if limit else clean["features"]
        clean_labels = clean["labels"][:limit] if limit else clean["labels"]
        report["backbones"][name] = {}

        for condition in _condition_list(cfg):
            if condition == "clean":
                continue
            path = cache_dir / f"{name}_{condition}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            features = data["features"][:limit] if limit else data["features"]
            logits = _head_logits(cfg, name, features, device)
            clean_logits = _head_logits(cfg, name, clean_features, device)
            metrics = classification_metrics(logits, clean_labels, num_classes)
            metrics["consistency"] = prediction_consistency(clean_logits, logits)
            report["conditions"].setdefault(condition, {})[name] = metrics
            report["stability"].setdefault(condition, {})[name] = {
                "cosine_clean_vs_transformed": cosine_stability(clean_features, features)}

            if clip_data is not None and name == "clip_vit_b32":
                text = clip_data["text_features"]
                scale = float(clip_data["scale"])
                zero_shot = scale * (features @ text.T)
                zero_clean = scale * (clean_features @ text.T)
                metrics_zero = classification_metrics(zero_shot, clean_labels, num_classes)
                metrics_zero["consistency"] = prediction_consistency(zero_clean, zero_shot)
                report["conditions"][condition][f"{name}_zero_shot"] = metrics_zero

        # translation curve: average over directions
        for pixels in TRANSLATION_DISPLACEMENTS:
            if pixels == 0:
                continue
            accuracies, consistencies, stabilities = [], [], []
            for direction in TRANSLATION_DIRECTIONS:
                condition = f"translate_{direction}_{pixels}"
                path = cache_dir / f"{name}_{condition}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                features = data["features"][:limit] if limit else data["features"]
                logits = _head_logits(cfg, name, features, device)
                clean_logits = _head_logits(cfg, name, clean_features, device)
                accuracies.append(classification_metrics(logits, clean_labels, num_classes)["accuracy"])
                consistencies.append(prediction_consistency(clean_logits, logits))
                stabilities.append(cosine_stability(clean_features, features))
            if accuracies:
                report["translation"].setdefault(name, {})[str(pixels)] = {
                    "accuracy": float(np.mean(accuracies)),
                    "consistency": float(np.mean(consistencies)),
                    "cosine_stability": float(np.mean(stabilities)),
                }

        # cue conflicts: head (trained) and CLIP zero-shot
        conflict_path = cache_dir / f"{name}_conflict.npz"
        if conflict_path.exists():
            data = np.load(conflict_path, allow_pickle=True)
            features = data["features"]
            content = data["content_labels"]
            style = data["style_labels"]
            head_predictions = _head_logits(cfg, name, features, device).argmax(axis=1)
            counts = shape_texture_counts(head_predictions, content, style)
            report["conflicts"].setdefault(name, {})["head"] = {
                **counts, **shape_bias_coverage(counts)}
            if clip_data is not None and name == "clip_vit_b32":
                zero = (float(clip_data["scale"]) * (features @ clip_data["text_features"].T)).argmax(axis=1)
                counts_zero = shape_texture_counts(zero, content, style)
                report["conflicts"][name]["zero_shot"] = {
                    **counts_zero, **shape_bias_coverage(counts_zero)}

        # representation visualisations for grayscale and patch shuffle
        for condition in ["grayscale", "shuffle"]:
            path = cache_dir / f"{name}_{condition}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            features = data["features"]
            n = min(300, len(clean_features), len(features))
            rng = np.random.RandomState(cfg["seed"])
            selection = rng.permutation(min(len(clean_features), len(features)))[:n]
            tsne_figure(clean_features[selection], features[selection], clean_labels[selection],
                        subsets["classes"], out_dir / f"tsne_{name}_{condition}.png",
                        title=f"{name}: clean vs {condition}", seed=cfg["seed"])

    # translation curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for name, curve in report["translation"].items():
            xs = sorted(int(k) for k in curve)
            axis.plot(xs, [curve[str(x)]["accuracy"] for x in xs], marker="o", label=f"{name} accuracy")
            axis.plot(xs, [curve[str(x)]["consistency"] for x in xs], marker="s", linestyle="--",
                      label=f"{name} consistency")
        axis.set_xlabel("displacement (px)")
        axis.set_ylabel("value")
        axis.set_title("translation curve (averaged over directions)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(out_dir / "translation_curve.png", dpi=150)
        plt.close(figure)
    except Exception as exc:
        log.warning("translation plot failed: %s", exc)

    with open(out_dir / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("wrote %s", out_dir / "analysis.json")


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data_root"] = args.data_root
    set_seed(cfg["seed"])
    logger = get_logger()
    logger.info("task1 stage=%s config_hash=%s", args.stage, config_hash(cfg))

    subsets = stage_prepare(cfg, args) if args.stage in ("prepare", "all", "cache", "heads", "analysis") else None
    if args.stage in ("conflicts", "all"):
        stage_conflicts(cfg, args)
    if args.stage in ("cache", "all"):
        stage_cache(cfg, args, subsets)
    if args.stage in ("heads", "all"):
        stage_heads(cfg, args, subsets)
    if args.stage in ("analysis", "all"):
        stage_analysis(cfg, args, subsets)


if __name__ == "__main__":
    main()
