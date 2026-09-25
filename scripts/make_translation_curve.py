"""Regenerate the Task 1 translation curve from the saved analysis.

``task1/run_task1.py`` produces this figure at the end of the Kaggle run; this
script rebuilds it from ``results/task1/analysis.json`` alone, so it can be
refreshed on CPU without the feature caches.  The rebuilt figure anchors every
predictor at 0 px (the clean image, i.e. the identity translation) and includes
the CLIP ViT-B/32 zero-shot series, which the original plot omitted.

    python scripts/make_translation_curve.py

Writes ``results/task1/translation_curve.png`` (committed artefact) and the
report copy ``doc_image/task1/translation_curve.png``.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "task1" / "analysis.json"
RESULTS_FIGURE = ROOT / "results" / "task1" / "translation_curve.png"
GALLERY_FIGURE = ROOT / "doc_image" / "task1" / "translation_curve.png"
REPORT_FIGURE = ROOT / "report" / "figures" / "task1_translation_curve.png"

DIRECTIONS = ["left", "right", "up", "down"]
PREDICTORS = ["resnet50", "vit_b16", "clip_vit_b32"]


def build_series(analysis: dict) -> dict:
    """Per-predictor {displacement: {accuracy, consistency}} including 0 px."""
    translation = analysis.get("translation", {})
    conditions = analysis["conditions"]
    series = {}

    for name in PREDICTORS:
        curve = {int(key): dict(values) for key, values in translation.get(name, {}).items()}
        clean = conditions["clean"][name]
        curve[0] = {"accuracy": float(clean["accuracy"]), "consistency": 1.0}
        series[name] = curve

    # CLIP zero-shot: the original plot is head-only, but zero-shot metrics
    # exist per condition; average them over directions like the heads.
    displacements = sorted({int(key) for curve in translation.values() for key in curve})
    zero_shot = {}
    for pixels in displacements:
        accuracies, consistencies = [], []
        for direction in DIRECTIONS:
            row = conditions.get(f"translate_{direction}_{pixels}", {}).get(
                "clip_vit_b32_zero_shot")
            if row:
                accuracies.append(row["accuracy"])
                consistencies.append(row["consistency"])
        if accuracies:
            zero_shot[pixels] = {
                "accuracy": float(np.mean(accuracies)),
                "consistency": float(np.mean(consistencies)),
            }
    clean_zero = conditions["clean"].get("clip_vit_b32_zero_shot")
    if clean_zero:
        zero_shot[0] = {"accuracy": float(clean_zero["accuracy"]), "consistency": 1.0}
    if zero_shot:
        series["clip_vit_b32_zero_shot"] = zero_shot
    return series


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
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    series = build_series(analysis)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for name, curve in series.items():
        xs = sorted(curve)
        label = name.replace("clip_vit_b32", "clip")
        axes[0].plot(xs, [curve[x]["accuracy"] for x in xs], marker="o", label=label)
        axes[1].plot(xs, [curve[x]["consistency"] for x in xs], marker="s", label=label)
    ticks = sorted({x for curve in series.values() for x in curve})
    for axis, title in zip(axes, ("accuracy", "prediction consistency")):
        axis.set_title(title)
        axis.set_xlabel("displacement (px)")
        axis.set_ylabel("value")
        axis.set_xticks(ticks)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("translation curve (averaged over directions; 0 px = clean)")
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    RESULTS_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(RESULTS_FIGURE, dpi=150)
    plt.close(figure)
    print("wrote", RESULTS_FIGURE.relative_to(ROOT))

    GALLERY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    _copy_with_retry(RESULTS_FIGURE, GALLERY_FIGURE)
    print("copied", GALLERY_FIGURE.relative_to(ROOT))

    REPORT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    _copy_with_retry(RESULTS_FIGURE, REPORT_FIGURE)
    print("copied", REPORT_FIGURE.relative_to(ROOT))


if __name__ == "__main__":
    main()
