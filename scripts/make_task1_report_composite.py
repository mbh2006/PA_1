"""Build the compact Task 1 report composite (five panels, one figure).

Panels
    (a) intervention accuracy (clean / grayscale / hue / shuffle)
    (b) cue-conflict shape bias and coverage
    (c) translation accuracy with the 0 px clean anchor
    (d) representation stability (grayscale / hue / shuffle / conflict)
    (e) three cue-conflict examples (content -> style)

Everything is rebuilt at the final physical size from
``results/task1/analysis.json`` and the conflict PNGs in ``report/figures/``
(no downscaled copies of the big gallery figures, so the labels stay legible).

    python scripts/make_task1_report_composite.py

Writes ``doc_image/task1/task1_composite.png`` and
``report/figures/task1_composite.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_translation_curve import build_series  # noqa: E402

ANALYSIS_PATH = ROOT / "results" / "task1" / "analysis.json"
CONFLICT_DIR = ROOT / "report" / "figures"
OUT_GALLERY = ROOT / "doc_image" / "task1" / "task1_composite.png"
OUT_REPORT = ROOT / "report" / "figures" / "task1_composite.png"

PREDICTORS = ["resnet50", "vit_b16", "clip_vit_b32", "clip_vit_b32_zero_shot"]
PREDICTOR_LABELS = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16",
                    "clip_vit_b32": "CLIP head", "clip_vit_b32_zero_shot": "CLIP zero-shot"}
BACKBONES = ["resnet50", "vit_b16", "clip_vit_b32"]
BACKBONE_LABELS = {"resnet50": "ResNet-50", "vit_b16": "ViT-B/16", "clip_vit_b32": "CLIP"}
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

CONFLICT_FILES = [
    "conflict_airplane_content_bird_style_000.png",
    "conflict_deer_content_dog_style_000.png",
    "conflict_ship_content_truck_style_000.png",
]


def conflict_label(stem: str) -> str:
    stem = Path(stem).stem
    content, _, rest = stem.replace("conflict_", "", 1).partition("_content_")
    style = rest.replace("_style_000", "")
    return f"{content} → {style}"


def main() -> None:
    import json

    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    conditions = analysis["conditions"]
    conflicts = analysis["conflicts"]
    stability = analysis["stability"]
    translation = build_series(analysis)

    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.5,
                         "axes.labelsize": 6.5, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "legend.fontsize": 5.5,
                         "figure.dpi": 200})

    figure = plt.figure(figsize=(7.16, 4.9))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85],
                               hspace=0.6, wspace=0.26,
                               left=0.065, right=0.99, top=0.90, bottom=0.075)

    # (a) intervention accuracy -------------------------------------------------
    axis = figure.add_subplot(grid[0, 0])
    base_conditions = ["clean", "grayscale", "hue", "shuffle"]
    x = np.arange(len(base_conditions))
    width = 0.2
    handles = []
    for index, name in enumerate(PREDICTORS):
        values = [conditions[condition][name]["accuracy"] for condition in base_conditions]
        bars = axis.bar(x + (index - 1.5) * width, values, width,
                        label=PREDICTOR_LABELS[name], color=COLORS[index])
        handles.append(bars[0])
    axis.set_title("(a) intervention accuracy")
    axis.set_xticks(x)
    axis.set_xticklabels(["clean", "gray", "hue", "shuffle"])
    axis.set_ylim(0.70, 1.0)
    axis.grid(axis="y", alpha=0.3)

    # (b) conflict shape bias / coverage ---------------------------------------
    axis = figure.add_subplot(grid[0, 1])

    def conflict_values(name):
        backbone = "clip_vit_b32" if name == "clip_vit_b32_zero_shot" else name
        predictor = "zero_shot" if name == "clip_vit_b32_zero_shot" else "head"
        return conflicts[backbone][predictor]

    x = np.arange(len(PREDICTORS))
    width = 0.38
    shape = [conflict_values(name)["shape_bias"] for name in PREDICTORS]
    coverage = [conflict_values(name)["coverage"] for name in PREDICTORS]
    axis.bar(x - width / 2, shape, width, label="shape bias", color="tab:blue")
    axis.bar(x + width / 2, coverage, width, label="coverage", color="tab:orange")
    axis.set_title("(b) cue conflicts: shape bias / coverage")
    axis.set_xticks(x)
    axis.set_xticklabels(["R-50", "ViT", "CLIP", "CLIP-zs"])
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower left", framealpha=0.9)

    # (c) translation accuracy --------------------------------------------------
    axis = figure.add_subplot(grid[1, 0])
    for index, name in enumerate(PREDICTORS):
        curve = translation[name]
        xs = sorted(curve)
        axis.plot(xs, [curve[value]["accuracy"] for value in xs], marker="o",
                  markersize=3, linewidth=1.1, color=COLORS[index],
                  label=PREDICTOR_LABELS[name])
    axis.set_title("(c) translation accuracy (0 px = clean)")
    axis.set_xticks([0, 8, 16, 32])
    axis.set_ylim(0.90, 0.995)
    axis.grid(alpha=0.3)

    # (d) representation stability ---------------------------------------------
    axis = figure.add_subplot(grid[1, 1])
    stability_conditions = ["grayscale", "hue", "shuffle", "conflict"]
    x = np.arange(len(stability_conditions))
    width = 0.26
    for index, name in enumerate(BACKBONES):
        values = [stability[condition][name]["cosine_clean_vs_transformed"]
                  for condition in stability_conditions]
        axis.bar(x + (index - 1) * width, values, width,
                 label=BACKBONE_LABELS[name], color=COLORS[index])
    axis.set_title("(d) representation stability (cosine)")
    axis.set_xticks(x)
    axis.set_xticklabels(["gray", "hue", "shuffle", "conflict"])
    axis.set_ylim(0, 0.95)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower left", framealpha=0.9)

    # (e) cue-conflict examples --------------------------------------------------
    strip = grid[2, :].subgridspec(1, 3, wspace=0.06)
    for index, filename in enumerate(CONFLICT_FILES):
        axis = figure.add_subplot(strip[0, index])
        image = plt.imread(CONFLICT_DIR / filename)
        axis.imshow(image)
        title = conflict_label(filename)
        axis.set_title(f"(e) {title}" if index == 0 else title, fontsize=6.5)
        axis.axis("off")

    # one shared legend for the predictor colours used in (a) and (c)
    figure.legend(handles, [PREDICTOR_LABELS[name] for name in PREDICTORS],
                  loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=4,
                  fontsize=5.5, frameon=False)

    OUT_GALLERY.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_GALLERY, dpi=200)
    plt.close(figure)
    print("wrote", OUT_GALLERY.relative_to(ROOT))

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(OUT_GALLERY, OUT_REPORT)
    print("copied", OUT_REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
