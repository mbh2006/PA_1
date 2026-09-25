"""Build the ``doc_image/`` gallery: one folder per task with every image needed
for the report.

It combines artefacts that already exist (training curves, t-SNE figures,
translation curve, ROC figure, example conflicts) with newly generated plots
from the committed result files:

* task1: dataset samples, intervention demo, conflict examples, accuracy /
  consistency by condition, shape bias and coverage, representation stability;
* task2: PACS domain samples, per-run curves, main metrics, per-class target
  accuracy, confusion matrices, domain separability;
* task3: per-run curves, source/Sketch metrics, source-domain separability,
  sharpness proxy;
* task4: CIFAR-10 and unknown sample grids, per-run curves, ROC figure, score
  distributions, OSR metric bars, accepted-failure gallery.

Only saved JSON/CSV/npz and local datasets are read - no training and no model
predictions are computed for the copied artefacts. The failure gallery recomputes
the score ranking from the cached logits (identical arithmetic to the committed
evaluation).

    python scripts/make_doc_images.py \
        --pacs-root "<PACS folder>" \
        --cache-dir "kaggle_outputs/t4_rest/PA_1/task4/cache" \
        --stl10-root data/stl10_dl
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
OUT = ROOT / "doc_image"

TASK2_RUNS = ["t2_source_only", "t2_dan", "t2_dann", "t2_cdan",
              "t2_dan_lambda01", "t2_dan_lambda10"]
TASK2_MAIN = [("t2_source_only", "Source-only"), ("t2_dan", "DAN"),
              ("t2_dann", "DANN"), ("t2_cdan", "CDAN")]
TASK3_RUNS = ["t3_dan_dg", "t3_sam", "t3_sam_rho001", "t3_sam_rho01"]
TASK4_RUNS = ["t4_vanilla", "t4_gcsc", "t4_proser"]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})


def load_json(path: Path):
    import json
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def source_mean_worst(metrics: Dict) -> Dict[str, float]:
    """Mean/worst source macro-F1; derived from per-domain values if absent.

    Task 3 runs store both values; the Task 2 ERM checkpoint reused as the
    Task 3 baseline stores only per-domain metrics, so the figure must derive
    them the same way the report table does.
    """
    if metrics.get("source_mean_macro_f1") is not None:
        return {"mean": metrics["source_mean_macro_f1"],
                "worst": metrics["source_worst_macro_f1"]}
    per_domain = [values["macro_f1"] for values in metrics["source_validation"].values()]
    return {"mean": sum(per_domain) / len(per_domain), "worst": min(per_domain)}


def _write_with_retry(write, *args, retries: int = 4, delay: float = 1.0, **kwargs) -> None:
    """Run ``write`` retrying on transient OSErrors (file-watchers, indexers)."""
    for attempt in range(retries):
        try:
            write(*args, **kwargs)
            return
        except OSError as exc:
            if attempt == retries - 1:
                raise
            print("retrying write:", exc)
            time.sleep(delay)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _write_with_retry(fig.savefig, path)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def copy_if(src: Path, dst: Path) -> None:
    src = Path(src)
    if not src.exists():
        print("skip (missing):", src.relative_to(ROOT) if src.is_relative_to(ROOT) else src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_with_retry(shutil.copy2, src, dst)
    print("copied", dst.relative_to(ROOT))


# --------------------------------------------------------------------------- task 1
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]


def load_stl10_train(stl10_root: Path):
    """Read the official STL-10 train split from its binary files.

    Mirrors ``torchvision.datasets.STL10`` loading but requires only
    ``stl10_binary/train_X.bin`` and ``train_y.bin`` (torchvision's integrity
    check wants the 2.7 GB unlabeled set too, which the report figures do not
    need).  Returns ``(images HWC uint8, labels 0-based, class names)``.
    """
    base = Path(stl10_root) / "stl10_binary"
    images = np.fromfile(str(base / "train_X.bin"), dtype=np.uint8)
    images = images.reshape(-1, 3, 96, 96).transpose(0, 1, 3, 2)
    labels = np.fromfile(str(base / "train_y.bin"), dtype=np.uint8).astype(np.int64) - 1
    names_file = base / "class_names.txt"
    classes = names_file.read_text().split() if names_file.exists() else STL10_CLASSES
    return images, labels, classes


def sample_grid_stl10(stl10_root: Path, out: Path) -> None:
    try:
        images, labels, classes = load_stl10_train(stl10_root)
    except Exception as exc:  # dataset not present
        print("skip STL-10 sample grid:", exc)
        return
    fig, axes = plt.subplots(3, 10, figsize=(16, 5.6))
    for column in range(10):
        indices = np.where(labels == column)[0][:3]
        for row in range(3):
            picture = images[int(indices[row])].transpose(1, 2, 0)
            axes[row, column].imshow(picture)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(classes[column], fontsize=9)
    fig.suptitle("STL-10: 3 samples per class (official train partition)")
    save(fig, out / "dataset_stl10_samples.png")


def intervention_demo(stl10_root: Path, out: Path) -> None:
    try:
        images, _, _ = load_stl10_train(stl10_root)
    except Exception as exc:
        print("skip intervention demo:", exc)
        return
    from PIL import Image
    from task1.data.interventions import grayscale, hue_rotate, patch_shuffle, translate
    image = Image.fromarray(images[0].transpose(1, 2, 0)).resize((224, 224))
    variants = [
        ("clean", image),
        ("grayscale", grayscale(image)),
        ("hue +120", hue_rotate(image, 120.0)),
        ("translate 32px right", translate(image, 32, "right")),
        ("patch shuffle", patch_shuffle(image, 0)),
    ]
    fig, axes = plt.subplots(1, len(variants), figsize=(15, 3.4))
    for axis, (title, picture) in zip(axes, variants):
        axis.imshow(np.asarray(picture))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    save(fig, out / "interventions_demo.png")


def conflict_examples(out: Path, examples_dir: Path) -> None:
    candidates = sorted(examples_dir.glob("conflict_*.png"))[:6]
    if not candidates:
        print("skip conflict examples: none found")
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, path in zip(axes.ravel(), candidates):
        axis.imshow(plt.imread(path))
        axis.axis("off")
        stem = path.stem.replace("conflict_", "")
        content, _, style = stem.partition("_content_")
        axis.set_title(f"content: {content} | style: {style.replace('_style', '')}", fontsize=9)
    save(fig, out / "conflict_examples.png")


def build_task1(out: Path, args) -> None:
    task_out = out / "task1"
    copy_if(RESULTS / "task1" / "translation_curve.png", task_out / "translation_curve.png")
    for figure in sorted((RESULTS / "task1").glob("tsne_*.png")):
        copy_if(figure, task_out / figure.name)
    conflict_examples(task_out, ROOT / "report" / "figures")
    sample_grid_stl10(Path(args.stl10_root), task_out)
    intervention_demo(Path(args.stl10_root), task_out)

    analysis = load_json(RESULTS / "task1" / "analysis.json")
    if not analysis:
        return
    models = ["resnet50", "vit_b16", "clip_vit_b32", "clip_vit_b32_zero_shot"]
    model_labels = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16",
                    "clip_vit_b32": "CLIP head", "clip_vit_b32_zero_shot": "CLIP zero-shot"}
    conditions = ["clean", "grayscale", "hue", "shuffle"]
    x = np.arange(len(conditions))
    width = 0.2
    for metric, filename, title, ylim in [
        ("accuracy", "conditions_accuracy.png", "Accuracy by condition", (0.7, 1.0)),
        ("macro_f1", "conditions_macro_f1.png", "Macro-F1 by condition", (0.7, 1.0)),
        ("mean_max_confidence", "conditions_confidence.png", "Mean max softmax by condition", (0.0, 1.0)),
        ("consistency", "conditions_consistency.png", "Prediction consistency vs clean", (0.7, 1.0)),
    ]:
        if not all(metric in analysis["conditions"][conditions[0]][m] for m in models):
            continue
        fig, axis = plt.subplots(figsize=(9, 4.4))
        for index, model in enumerate(models):
            values = [analysis["conditions"][condition][model].get(metric, np.nan)
                      for condition in conditions]
            axis.bar(x + (index - 1.5) * width, values, width, label=model_labels[model])
        axis.set_xticks(x)
        axis.set_xticklabels(conditions)
        axis.set_ylim(*ylim)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)
        save(fig, task_out / filename)

    stability_conditions = ["grayscale", "hue", "shuffle", "conflict"]
    backbones = ["resnet50", "vit_b16", "clip_vit_b32"]
    x = np.arange(len(stability_conditions))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 4.4))
    for index, backbone in enumerate(backbones):
        values = [analysis["stability"].get(condition, {}).get(backbone, {}).get(
            "cosine_clean_vs_transformed", np.nan) for condition in stability_conditions]
        axis.bar(x + (index - 1) * width, values, width, label=model_labels[backbone])
    axis.set_xticks(x)
    axis.set_xticklabels(stability_conditions)
    axis.set_ylim(0, 1)
    axis.set_title("Representation cosine stability vs clean (conflict: vs content image)")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    save(fig, task_out / "representation_stability.png")

    entries: List[Tuple[str, Dict]] = []
    for backbone, values in analysis["conflicts"].items():
        for predictor, metrics in values.items():
            name = model_labels.get(backbone, backbone)
            if predictor == "zero_shot":
                name += " (zero-shot)"
            entries.append((name, metrics))
    x = np.arange(len(entries))
    width = 0.35
    fig, axis = plt.subplots(figsize=(9, 4.4))
    axis.bar(x - width / 2, [e[1]["shape_bias"] for e in entries], width, label="shape bias (%)")
    axis.bar(x + width / 2, [e[1]["coverage"] for e in entries], width, label="coverage (%)")
    axis.set_xticks(x)
    axis.set_xticklabels([e[0] for e in entries], fontsize=8)
    axis.set_ylim(0, 100)
    axis.set_title("Cue conflicts: shape bias and coverage (265 conflicts)")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    save(fig, task_out / "conflict_bias_coverage.png")


# --------------------------------------------------------------------------- task 2
def pacs_sample_grid(pacs_root: Path, out: Path, title: str = "PACS") -> None:
    root = Path(pacs_root)
    if not root.exists():
        print("skip PACS sample grid:", root)
        return
    classes = ["dog", "elephant", "horse", "person"]
    domains = ["photo", "art_painting", "cartoon", "sketch"]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for row, domain in enumerate(domains):
        for column, class_name in enumerate(classes):
            images = sorted((root / domain / class_name).glob("*.jpg"))
            axis = axes[row, column]
            if images:
                axis.imshow(plt.imread(images[0]))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(class_name, fontsize=11)
            if column == 0:
                axis.set_ylabel(domain, fontsize=10)
    fig.suptitle(f"{title}: rows = domains, columns = classes")
    save(fig, out / "dataset_pacs_domains.png")


def confusion_figure(rows: Sequence[Tuple[str, Dict]], out: Path, class_names: Sequence[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for axis, (label, metrics) in zip(axes.ravel(), rows):
        matrix = np.array(metrics["target"]["confusion_matrix"], dtype=float)
        matrix = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        axis.set_yticklabels(class_names, fontsize=8)
        axis.set_title(f"{label}: target confusion (row-normalised)")
        fig.colorbar(image, ax=axis, fraction=0.046)
    save(fig, out / "target_confusions.png")


def build_task2(out: Path, args) -> None:
    task_out = out / "task2"
    pacs_sample_grid(Path(args.pacs_root), task_out)
    for run in TASK2_RUNS:
        for figure in ("curves_train.png", "curves_val.png"):
            copy_if(RESULTS / run / figure, task_out / f"{run}_{figure}")

    metrics = {run: load_json(RESULTS / run / "final_metrics.json") for run, _ in TASK2_MAIN}
    metrics = {run: value for run, value in metrics.items() if value}
    if not metrics:
        return
    labels = [label for run, label in TASK2_MAIN if run in metrics]
    source_mean = [float(np.mean([m["source_validation"][d]["macro_f1"]
                                  for d in m["source_validation"]])) for m in metrics.values()]
    target_acc = [m["target"]["accuracy"] for m in metrics.values()]
    target_f1 = [m["target"]["macro_f1"] for m in metrics.values()]
    separability = []
    for run in metrics:
        data = load_json(RESULTS / run / "domain_separability.json")
        separability.append(data["score"] if data else np.nan)

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].bar(x, source_mean, 0.5)
    axes[0].set_title("Source validation mean macro-F1")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x - width / 2, target_acc, width, label="accuracy")
    axes[1].bar(x + width / 2, target_f1, width, label="macro-F1")
    axes[1].set_title("Sketch target accuracy / macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[2].bar(x, separability, 0.5, color="tab:red")
    axes[2].axhline(0.5, color="k", linestyle="--", linewidth=1, label="chance")
    axes[2].set_title("Domain separability (chance 0.5)")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.3)
    save(fig, task_out / "main_metrics.png")

    from shared.pacs import CLASSES
    fig, axis = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(CLASSES))
    width = 0.2
    for index, (run, label) in enumerate((run, label) for run, label in TASK2_MAIN if run in metrics):
        per_class = {row["class"]: row["accuracy"] for row in metrics[run]["target_per_class"]}
        axis.bar(x + (index - 1.5) * width, [per_class.get(c, np.nan) for c in CLASSES],
                 width, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(CLASSES, rotation=20)
    axis.set_ylim(0, 1)
    axis.set_title("Sketch per-class accuracy by method")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    save(fig, task_out / "target_per_class.png")

    confusion_figure([(label, metrics[run]) for run, label in TASK2_MAIN if run in metrics],
                     task_out, CLASSES)


# --------------------------------------------------------------------------- task 3
def build_task3(out: Path, args) -> None:
    task_out = out / "task3"
    for run in TASK3_RUNS:
        for figure in ("curves_train.png", "curves_val.png"):
            copy_if(RESULTS / run / figure, task_out / f"{run}_{figure}")

    entries: List[Tuple[str, Optional[Dict], Path]] = [
        ("ERM", load_json(RESULTS / "t2_source_only" / "final_metrics.json"), RESULTS / "t3_erm"),
        ("DAN-DG", load_json(RESULTS / "t3_dan_dg" / "final_metrics.json"), RESULTS / "t3_dan_dg"),
        ("SAM", load_json(RESULTS / "t3_sam" / "final_metrics.json"), RESULTS / "t3_sam"),
    ]
    entries = [entry for entry in entries if entry[1]]
    if not entries:
        return
    labels = [entry[0] for entry in entries]
    source_stats = [source_mean_worst(entry[1]) for entry in entries]
    mean_source = [stats["mean"] for stats in source_stats]
    worst_source = [stats["worst"] for stats in source_stats]
    sketch_acc = [entry[1]["target"]["accuracy"] for entry in entries]
    sketch_f1 = [entry[1]["target"]["macro_f1"] for entry in entries]
    separability = []
    sharpness = []
    for _, _, run_dir in entries:
        data = load_json(run_dir / "source_domain_separability.json")
        separability.append(data["score"] if data else np.nan)
        data = load_json(run_dir / "sharpness.json")
        sharpness.append(data["sharpness_delta"] if data else np.nan)

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    axes[0].bar(x - width / 2, mean_source, width, label="mean")
    axes[0].bar(x + width / 2, worst_source, width, label="worst")
    axes[0].set_title("Source validation macro-F1")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[1].bar(x - width / 2, sketch_acc, width, label="accuracy")
    axes[1].bar(x + width / 2, sketch_f1, width, label="macro-F1")
    axes[1].set_title("Sketch (unseen) accuracy / macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[2].bar(x, separability, 0.5, color="tab:red")
    axes[2].axhline(1 / 3, color="k", linestyle="--", linewidth=1, label="chance (1/3)")
    axes[2].set_title("Source-domain separability")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=8)
    sharpness_values = np.array(sharpness, dtype=float)
    axes[3].bar(x, sharpness_values, 0.5, color="tab:green")
    axes[3].set_yscale("log")
    axes[3].set_title("Sharpness proxy (log scale)")
    for index, value in enumerate(sharpness_values):
        axes[3].text(index, value * 1.15, f"{value:.3g}", ha="center", fontsize=8)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8)
        axis.grid(axis="y", alpha=0.3)
    save(fig, task_out / "main_metrics_and_diagnostics.png")


# --------------------------------------------------------------------------- task 4
def cifar10_sample_grid(data_root: Path, out: Path) -> None:
    try:
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=str(data_root), train=True, download=False)
    except Exception as exc:
        print("skip CIFAR-10 sample grid:", exc)
        return
    targets = np.array(dataset.targets)
    fig, axes = plt.subplots(3, 10, figsize=(16, 5.6))
    for column in range(10):
        indices = np.where(targets == column)[0][:3]
        for row in range(3):
            image, _ = dataset[int(indices[row])]
            axes[row, column].imshow(np.asarray(image))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(dataset.classes[column], fontsize=8)
    fig.suptitle("CIFAR-10: 3 samples per known class")
    save(fig, out / "dataset_cifar10_samples.png")


def cifar100_unknown_grid(data_root: Path, out: Path) -> None:
    try:
        from task4.data.cifar100_unknowns import FAR_UNKNOWN, NEAR_UNKNOWN, Cifar100Unknowns
        near = Cifar100Unknowns(str(data_root), "near", None)
        far = Cifar100Unknowns(str(data_root), "far", None)
    except Exception as exc:
        print("skip CIFAR-100 unknown grid:", exc)
        return
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for block, (dataset, names, row0) in enumerate([(near, NEAR_UNKNOWN, 0), (far, FAR_UNKNOWN, 2)]):
        for column, class_name in enumerate(names):
            for k in range(2):
                fine_label = dataset.fine_labels[column]
                positions = [i for i, index in enumerate(dataset.indices)
                             if dataset.base.targets[index] == fine_label]
                if k >= len(positions):
                    continue
                image, _ = dataset.base[dataset.indices[positions[k]]]
                axis = axes[row0 + k, column]
                axis.imshow(np.asarray(image))
                axis.axis("off")
                if k == 0:
                    axis.set_title(class_name, fontsize=8)
    axes[0, 0].annotate("NEAR", xy=(-0.35, 0.5), xycoords="axes fraction", rotation=90,
                        fontsize=12, va="center")
    axes[2, 0].annotate("FAR", xy=(-0.35, 0.5), xycoords="axes fraction", rotation=90,
                        fontsize=12, va="center")
    fig.suptitle("CIFAR-100 unknowns (evaluation only): 2 samples per class")
    save(fig, out / "dataset_cifar100_unknowns.png")


def score_distributions(cache_dir: Path, out: Path) -> None:
    path = cache_dir / "t4_vanilla.npz"
    if not path.exists():
        print("skip score distributions: cache missing")
        return
    from task4.evaluation.metrics import operating_point
    from task4.scores.energy import energy
    from task4.scores.mahalanobis import fit_mahalanobis, mahalanobis
    from task4.scores.mls import mls
    from task4.scores.msp import msp

    with np.load(path, allow_pickle=True) as data:
        cache = {key: data[key] for key in data.files}
    known = int(cache["num_known"])
    means, covariance = fit_mahalanobis(cache["features_train"], cache["labels_train"], known)
    score_functions = [
        ("MSP", lambda split: msp(cache[f"logits_{split}"][:, :known])),
        ("MLS", lambda split: mls(cache[f"logits_{split}"][:, :known])),
        ("Energy", lambda split: energy(cache[f"logits_{split}"][:, :known])),
        ("Mahalanobis", lambda split: mahalanobis(cache[f"features_{split}"], means, covariance)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (name, function) in zip(axes.ravel(), score_functions):
        values = {split: function(split) for split in ("val", "test", "near", "far")}
        operating = operating_point(values["val"], values["test"], values["near"])
        bins = np.histogram_bin_edges(np.concatenate([values["test"], values["near"], values["far"]]),
                                      bins=60)
        axis.hist(values["test"], bins=bins, density=True, alpha=0.45, label="known (test)")
        axis.hist(values["near"], bins=bins, density=True, alpha=0.45, label="near unknown")
        axis.hist(values["far"], bins=bins, density=True, alpha=0.45, label="far unknown")
        axis.axvline(operating["threshold"], color="k", linestyle="--", linewidth=1.2,
                     label=f"tau (95th pct) = {operating['threshold']:.3g}")
        axis.set_title(f"{name}: score distributions and calibrated threshold")
        axis.set_ylabel("density")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    save(fig, out / "score_distributions.png")


def osr_metric_bars(cache_dir: Path, out: Path) -> None:
    metrics = load_json(RESULTS / "t4_osr" / "osr_metrics.json")
    if not metrics:
        return
    rows = metrics["models"]
    labels = [f"{row['model']}\n{row['score']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - width / 2, [row["auroc_near"] for row in rows], width, label="near")
    axes[0].bar(x + width / 2, [row["auroc_far"] for row in rows], width, label="far")
    axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("AUROC (unknown as positive)")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=8)
    axes[1].bar(x - width / 2, [row["near_rejection"] for row in rows], width, label="near")
    axes[1].bar(x + width / 2, [row["far_rejection"] for row in rows], width, label="far")
    axes[1].set_title("Unknown rejection at the calibrated threshold")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8)
        axis.grid(axis="y", alpha=0.3)
    save(fig, out / "osr_metric_bars.png")

    posthoc = metrics["posthoc_vanilla"]
    labels = [row["score"] for row in posthoc]
    x = np.arange(len(posthoc))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - width / 2, [row["auroc_near"] for row in posthoc], width, label="near")
    axes[0].bar(x + width / 2, [row["auroc_far"] for row in posthoc], width, label="far")
    axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("Vanilla model: post-hoc score AUROC")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=8)
    axes[1].bar(x - width / 2, [row["near_rejection"] for row in posthoc], width, label="near")
    axes[1].bar(x + width / 2, [row["far_rejection"] for row in posthoc], width, label="far")
    axes[1].set_title("Vanilla model: rejection at threshold")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8)
        axis.grid(axis="y", alpha=0.3)
    save(fig, out / "posthoc_score_bars.png")


def _verify_failure_gallery(figure, image_axes, label_axes, text_axes, expected_texts,
                            tau_text) -> None:
    """Programmatic pre-save checks for the failure gallery (raises on failure)."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    problems = []

    def extent(artist):
        return artist.get_window_extent(renderer=renderer)

    # 1. no text may be clipped outside the canvas
    for text in figure.findobj(match=plt.Text):
        box = extent(text)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            problems.append(f"text outside canvas: {text.get_text()!r}")

    # 2. row labels must not overlap any image axes
    for label_axis in label_axes:
        label_box = extent(label_axis)
        for image_axis in image_axes:
            image_box = extent(image_axis)
            overlap_x = min(label_box.x1, image_box.x1) - max(label_box.x0, image_box.x0)
            overlap_y = min(label_box.y1, image_box.y1) - max(label_box.y0, image_box.y0)
            if overlap_x > 0 and overlap_y > 0:
                problems.append("row label overlaps an image axes")

    # 3. row labels vertically centred on their image row
    for row, label_axis in enumerate(label_axes):
        row_axes = image_axes[3 * row:3 * row + 3]
        row_centre = sum(extent(axis).y0 + extent(axis).height / 2 for axis in row_axes) / 3
        label_centre = extent(label_axis).y0 + extent(label_axis).height / 2
        if abs(row_centre - label_centre) > 2:
            problems.append(f"row label {row} is not centred on its image row")

    # 4. near examples in the top row, far examples below
    near_centre = sum(extent(axis).y0 for axis in image_axes[:3]) / 3
    far_centre = sum(extent(axis).y0 for axis in image_axes[3:]) / 3
    if near_centre <= far_centre:
        problems.append("near examples are not in the top row")

    # 5. displayed labels match the displayed samples; tau appears in every score
    actual = [tuple(text.get_text() for text in axis.texts) for axis in text_axes]
    if actual != [tuple(pair) for pair in expected_texts]:
        problems.append("panel labels do not match the displayed samples")
    for _, score_text in expected_texts:
        if tau_text not in score_text:
            problems.append(f"score line missing the calibrated tau: {score_text!r}")

    # 6. all image panels have identical dimensions
    boxes = [extent(axis) for axis in image_axes]
    for box in boxes[1:]:
        if abs(box.width - boxes[0].width) > 0.5 or abs(box.height - boxes[0].height) > 0.5:
            problems.append("image panels do not have identical dimensions")

    # 7. panel texts must not collide with each other
    panel_texts = [text for axis in text_axes for text in axis.texts]
    for i, first in enumerate(panel_texts):
        for second in panel_texts[i + 1:]:
            box_a, box_b = extent(first), extent(second)
            overlap_x = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            overlap_y = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if overlap_x > 0 and overlap_y > 0:
                problems.append(f"panel texts overlap: {first.get_text()!r}")

    if problems:
        raise SystemExit("failure gallery layout check failed:\n  " + "\n  ".join(problems))
    print(f"failure gallery layout checks: OK ({len(image_axes)} image panels, "
          f"{len(label_axes)} row labels)")


def failure_gallery(cache_dir: Path, data_root: Path, out: Path) -> None:
    """Accepted-unknown gallery, 2x3 (near unknowns top, far unknowns bottom).

    Visualization only: samples, predictions, scores, threshold and ordering are
    identical to the evaluation. The layout uses a dedicated row-label column and
    a text band above each image row, so class titles and score lines can never
    resize or overlap the images.
    """
    path = cache_dir / "t4_vanilla.npz"
    if not path.exists():
        print("skip failure gallery: cache missing")
        return
    from task4.data.cifar10 import CIFAR10_CLASSES
    from task4.data.cifar100_unknowns import Cifar100Unknowns
    from task4.evaluation.thresholds import threshold_at_percentile
    from task4.scores.mls import mls

    with np.load(path, allow_pickle=True) as data:
        cache = {key: data[key] for key in data.files}
    known = int(cache["num_known"])
    tau = float(threshold_at_percentile(mls(cache["logits_val"][:, :known]), 95.0))

    def signed(value: float) -> str:
        return ("−" if value < 0 else "") + f"{abs(value):.2f}"

    tau_text = f"τ = {signed(tau)}"
    groups = [("near", "Near unknowns"), ("far", "Far unknowns")]

    # explicit margins via GridSpec (never tight_layout): one narrow label column
    # and a text band above each image row
    figure = plt.figure(figsize=(7.0, 4.8))
    grid = figure.add_gridspec(
        4, 4, width_ratios=[0.6, 1.0, 1.0, 1.0], height_ratios=[0.50, 1.0, 0.50, 1.0],
        left=0.03, right=0.975, top=0.90, bottom=0.085, wspace=0.25, hspace=0.12)

    image_axes, label_axes, text_axes, expected_texts = [], [], [], []
    for row, (group, row_label) in enumerate(groups):
        dataset = Cifar100Unknowns(str(data_root), group, None)
        logits = cache[f"logits_{group}"][:, :known]
        unknownness = mls(logits)
        accepted = np.where(unknownness <= tau)[0]
        order = accepted[np.argsort(unknownness[accepted])][:3]

        label_axis = figure.add_subplot(grid[2 * row + 1, 0])
        label_axis.axis("off")
        label_axis.text(0.5, 0.5, row_label, ha="center", va="center",
                        fontsize=11, fontweight="bold", rotation=0)
        label_axes.append(label_axis)

        for column, position in enumerate(order):
            image, fine_label = dataset.base[dataset.indices[position]]
            predicted = int(logits[position].argmax())
            class_text = (f"{dataset.label_names[fine_label].replace('_', ' ')}"
                          f" → {CIFAR10_CLASSES[predicted]}")
            score_text = f"u = {signed(float(unknownness[position]))} ≤ {tau_text}"

            text_axis = figure.add_subplot(grid[2 * row, column + 1])
            text_axis.axis("off")
            text_axis.text(0.5, 0.72, class_text, ha="center", va="center",
                           fontsize=12, fontweight="bold")
            text_axis.text(0.5, 0.20, score_text, ha="center", va="center",
                           fontsize=10.5, color="#333333")
            text_axes.append(text_axis)
            expected_texts.append((class_text, score_text))

            axis = figure.add_subplot(grid[2 * row + 1, column + 1])
            axis.imshow(np.asarray(image))
            axis.axis("off")
            image_axes.append(axis)

    figure.suptitle("Confidently accepted unknowns — Vanilla, MLS",
                    fontsize=13, fontweight="bold", x=0.5, y=0.965)
    figure.text(0.5, 0.025,
                "Lower u = more confidently accepted; τ is calibrated on known "
                "validation data.", ha="center", fontsize=9)

    _verify_failure_gallery(figure, image_axes, label_axes, text_axes,
                            expected_texts, tau_text)

    out_path = out / "failure_gallery.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_with_retry(figure.savefig, out_path, dpi=300)
    plt.close(figure)
    print("wrote", out_path.relative_to(ROOT))
    copy_if(out_path, ROOT / "report" / "figures" / "task4_failure_gallery.png")


def build_task4(out: Path, args) -> None:
    task_out = out / "task4"
    for run in TASK4_RUNS:
        for figure in ("curves_loss.png", "curves_val.png"):
            copy_if(RESULTS / run / figure, task_out / f"{run}_{figure}")
    copy_if(RESULTS / "t4_osr" / "roc_scores.png", task_out / "roc_scores.png")
    cifar10_sample_grid(Path(args.data_root), task_out)
    cifar100_unknown_grid(Path(args.data_root), task_out)
    cache_dir = Path(args.cache_dir)
    score_distributions(cache_dir, task_out)
    osr_metric_bars(cache_dir, task_out)
    failure_gallery(cache_dir, Path(args.data_root), task_out)


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--pacs-root", default="data/pacs")
    parser.add_argument("--stl10-root", default="data/stl10_dl")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="kaggle_outputs/t4_rest/PA_1/task4/cache")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    build_task1(out, args)
    build_task2(out, args)
    build_task3(out, args)
    build_task4(out, args)
    print("\ndoc_image ready at", out)


if __name__ == "__main__":
    main()
