"""Build the compact Task 1 qualitative panel (compact t-SNE + conflict examples).

Top row: clean vs shuffled t-SNE for ResNet-50, ViT-B/16 and CLIP (regenerated
at final size from the feature caches, matching the protocol of the full-size
figures: 300 paired images, seed 6304, perplexity min(30, n/4), init pca).
Bottom row: three cue-conflict examples (content -> style).

    python scripts/make_task1_qualitative_panel.py \
        --cache-dir kaggle_outputs/t1_task1/PA_1/task1/cache

The cache directory is the one pulled from the Kaggle run (contains
``{backbone}_{clean,shuffle}.npz``); it is gitignored, so pass the path or keep
the default.  Writes ``doc_image/task1/task1_qualitative_panel.png`` and
``report/figures/task1_qualitative_panel.png``.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "kaggle_outputs" / "t1_task1" / "PA_1" / "task1" / "cache"
CONFLICT_DIR = ROOT / "report" / "figures"
OUT_GALLERY = ROOT / "doc_image" / "task1" / "task1_qualitative_panel.png"
OUT_REPORT = ROOT / "report" / "figures" / "task1_qualitative_panel.png"

BACKBONES = ["resnet50", "vit_b16", "clip_vit_b32"]
BACKBONE_TITLES = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "clip_vit_b32": "CLIP"}
CLASS_NAMES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey",
               "ship", "truck"]
CONFLICT_FILES = [
    "conflict_airplane_content_bird_style_000.png",
    "conflict_deer_content_dog_style_000.png",
    "conflict_ship_content_truck_style_000.png",
]
SEED = 6304


def conflict_label(filename: str) -> str:
    stem = Path(filename).stem
    content, _, rest = stem.replace("conflict_", "", 1).partition("_content_")
    return f"{content} → {rest.replace('_style_000', '')}"


def tsne_embedding(cache_dir: Path, backbone: str):
    clean = np.load(cache_dir / f"{backbone}_clean.npz")
    shuffled = np.load(cache_dir / f"{backbone}_shuffle.npz")
    from sklearn.manifold import TSNE

    rng = np.random.RandomState(SEED)
    n = min(300, len(clean["features"]), len(shuffled["features"]))
    selection = rng.permutation(min(len(clean["features"]), len(shuffled["features"])))[:n]
    combined = np.vstack([clean["features"][selection], shuffled["features"][selection]]
                         ).astype(np.float64)
    labels = clean["labels"][selection]
    tsne = TSNE(n_components=2, random_state=SEED,
                perplexity=min(30, len(combined) // 4), init="pca",
                learning_rate="auto")
    return tsne.fit_transform(combined), labels, n


def _copy_with_retry(source: Path, destination: Path, retries: int = 4,
                     delay: float = 1.0) -> None:
    """Copy retrying on transient OSErrors (file watchers, indexers)."""
    for attempt in range(retries):
        try:
            shutil.copy2(source, destination)
            return
        except OSError as exc:
            if attempt == retries - 1:
                raise
            print("retrying copy:", exc)
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    if not (cache_dir / "resnet50_shuffle.npz").exists():
        raise SystemExit(f"feature cache not found at {cache_dir}; pass --cache-dir")

    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "figure.dpi": 200})
    figure = plt.figure(figsize=(7.16, 4.5))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.5, 0.72],
                               hspace=0.28, wspace=0.16,
                               left=0.045, right=0.985, top=0.85, bottom=0.05)

    for column, backbone in enumerate(BACKBONES):
        embedding, labels, n = tsne_embedding(cache_dir, backbone)
        axis = figure.add_subplot(grid[0, column])
        colour_map = plt.get_cmap("tab10")
        for label in sorted(set(int(value) for value in labels)):
            mask = np.asarray(labels) == label
            axis.scatter(embedding[:n][mask, 0], embedding[:n][mask, 1], s=5,
                         color=colour_map(label % 10), alpha=0.7, linewidths=0)
            axis.scatter(embedding[n:][mask, 0], embedding[n:][mask, 1], s=7,
                         color=colour_map(label % 10), alpha=0.7, marker="x",
                         linewidths=0.7)
        title = BACKBONE_TITLES[backbone]
        axis.set_title(f"(a) {title}" if column == 0 else title)
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            handles = [Line2D([], [], marker="o", linestyle="", color="grey",
                              markersize=4, label="clean"),
                       Line2D([], [], marker="x", linestyle="", color="grey",
                              markersize=4, label="shuffled")]
            axis.legend(handles=handles, loc="lower left", fontsize=5,
                        framealpha=0.9, borderpad=0.3)

    for column, filename in enumerate(CONFLICT_FILES):
        axis = figure.add_subplot(grid[1, column])
        axis.imshow(plt.imread(CONFLICT_DIR / filename))
        label = conflict_label(filename)
        axis.set_title(f"(b) {label}" if column == 0 else label)
        axis.axis("off")

    class_handles = [Patch(facecolor=plt.get_cmap("tab10")(index % 10), label=name)
                     for index, name in enumerate(CLASS_NAMES)]
    figure.legend(handles=class_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncols=10, fontsize=5, frameon=False, handlelength=0.8,
                  columnspacing=0.7, handletextpad=0.3)

    OUT_GALLERY.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_GALLERY, dpi=200)
    plt.close(figure)
    print("wrote", OUT_GALLERY.relative_to(ROOT))

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _copy_with_retry(OUT_GALLERY, OUT_REPORT)
    print("copied", OUT_REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
