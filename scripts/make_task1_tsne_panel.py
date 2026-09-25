"""Compact three-panel t-SNE figure (clean vs patch-shuffled) for the report.

Panels: ResNet-50, ViT-B/16 and CLIP under the same clean-vs-shuffle comparison.
There is exactly one shared condition legend (top) and one shared class legend
(bottom), so no panel repeats either legend and the scatter panels can be larger.

Identical protocol to the full-size panels: 300 paired images from the 500-image
test subset, seed 6304, perplexity min(30, n/4), init PCA; circles = clean,
crosses = shuffled, colour = ground-truth class.

    python scripts/make_task1_tsne_panel.py [--cache-dir <dir>]

The cache directory is the one pulled from the Kaggle run
(``{backbone}_{clean,shuffle}.npz``); it is gitignored, so pass the path or keep
the default.  Writes ``doc_image/task1/task1_tsne_panel.png`` and
``report/figures/task1_tsne_panel.png``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_task1_qualitative_panel import (  # noqa: E402
    BACKBONE_TITLES, BACKBONES, CLASS_NAMES, DEFAULT_CACHE, tsne_embedding)

OUT_GALLERY = ROOT / "doc_image" / "task1" / "task1_tsne_panel.png"
OUT_REPORT = ROOT / "report" / "figures" / "task1_tsne_panel.png"


def _copy_with_retry(source: Path, destination: Path, retries: int = 4,
                     delay: float = 1.0) -> None:
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

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8.5, "figure.dpi": 200})
    figure = plt.figure(figsize=(7.0, 3.1))
    grid = figure.add_gridspec(1, 3, left=0.012, right=0.988, top=0.845,
                               bottom=0.135, wspace=0.07)

    colour_map = plt.get_cmap("tab10")
    for column, backbone in enumerate(BACKBONES):
        embedding, labels, n = tsne_embedding(cache_dir, backbone)
        axis = figure.add_subplot(grid[0, column])
        for label in sorted(set(int(value) for value in labels)):
            mask = np.asarray(labels) == label
            axis.scatter(embedding[:n][mask, 0], embedding[:n][mask, 1], s=7,
                         color=colour_map(label % 10), alpha=0.75, linewidths=0)
            axis.scatter(embedding[n:][mask, 0], embedding[n:][mask, 1], s=9,
                         color=colour_map(label % 10), alpha=0.75, marker="x",
                         linewidths=0.8)
        axis.set_title(BACKBONE_TITLES[backbone])
        axis.set_xticks([])
        axis.set_yticks([])

    # one shared condition legend for the whole figure
    condition_handles = [
        Line2D([], [], marker="o", linestyle="", color="#555555", markersize=5,
               label="clean"),
        Line2D([], [], marker="x", linestyle="", color="#555555", markersize=5,
               markeredgewidth=1.0, label="shuffled")]
    figure.legend(handles=condition_handles, loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=7.5, frameon=False)

    # one shared class legend across the bottom
    class_handles = [Patch(facecolor=colour_map(index % 10), label=name)
                     for index, name in enumerate(CLASS_NAMES)]
    figure.legend(handles=class_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.005), ncols=10, fontsize=6.5,
                  frameon=False, handlelength=0.9, columnspacing=0.9,
                  handletextpad=0.35)

    OUT_GALLERY.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_GALLERY, dpi=300)
    plt.close(figure)
    print("wrote", OUT_GALLERY.relative_to(ROOT))

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _copy_with_retry(OUT_GALLERY, OUT_REPORT)
    print("copied", OUT_REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
