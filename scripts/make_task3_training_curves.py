"""Compact Task 3 training-dynamics figure (one panel per method).

    (a) ERM    : classification loss (Task 2 source-only run reused unchanged)
    (b) DAN-DG : classification loss + pairwise source MMD
    (c) SAM    : classification loss

Solid curves use the left axis (classification loss, shared 0-0.8 range so the
panels are comparable); the dashed curve in (b) uses the right axis (MMD).
Values come straight from the run histories (``results/*/history.csv``), no
recomputation.

    python scripts/make_task3_training_curves.py

Writes ``doc_image/task3/task3_training_curves.png`` and
``report/figures/task3_training_curves.png``.
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("t2_source_only", "ERM", None),
    ("t3_dan_dg", "DAN-DG", "pairwise MMD"),
    ("t3_sam", "SAM", None),
]
OUT_GALLERY = ROOT / "doc_image" / "task3" / "task3_training_curves.png"
OUT_REPORT = ROOT / "report" / "figures" / "task3_training_curves.png"

CLS_COLOR = "tab:blue"
SECONDARY_COLOR = "tab:orange"
LEFT_LIMITS = (0.0, 0.8)


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


def load_history(run: str):
    with open(ROOT / "results" / run / "history.csv", "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def column(name):
        return [float(row[name]) for row in rows if row[name] not in ("", "nan")]

    history = {
        "epochs": [int(row["epoch"]) for row in rows],
        "cls": column("train_cls_loss"),
    }
    history["mmd"] = column("train_mmd") if "train_mmd" in rows[0] else []
    return history


def tick_labels(epochs):
    last = epochs[-1]
    return [str(value) if value == 1 or value == last or (value - 1) % 4 == 0 else ""
            for value in epochs]


def main() -> None:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8.5,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "figure.dpi": 200})

    figure = plt.figure(figsize=(7.0, 2.6))
    grid = figure.add_gridspec(1, 3, left=0.06, right=0.975, top=0.84,
                               bottom=0.185, wspace=0.56)

    for column, (run, title, secondary_label) in enumerate(RUNS):
        history = load_history(run)
        print(f"{title:7s} epochs {len(history['epochs']):2d}  "
              f"cls {history['cls'][0]:.4f} -> {history['cls'][-1]:.4f}"
              + (f"  mmd {history['mmd'][0]:.4f} -> {history['mmd'][-1]:.4f}"
                 if history["mmd"] else ""))

        primary = figure.add_subplot(grid[0, column])
        primary.plot(history["epochs"], history["cls"], color=CLS_COLOR,
                     marker="o", markersize=2.6, linewidth=1.2)
        primary.set_title(title)
        primary.set_xlabel("epoch")
        primary.set_xticks(history["epochs"])
        primary.set_xticklabels(tick_labels(history["epochs"]))
        primary.set_xlim(0.5, history["epochs"][-1] + 0.5)
        primary.set_ylim(*LEFT_LIMITS)
        primary.grid(axis="y", alpha=0.25)
        primary.tick_params(axis="y", colors=CLS_COLOR)
        if column == 0:
            primary.set_ylabel("classification loss", color=CLS_COLOR)

        if history["mmd"]:
            secondary = primary.twinx()
            secondary.plot(history["epochs"][:len(history["mmd"])], history["mmd"],
                           color=SECONDARY_COLOR, linestyle="--", marker="s",
                           markersize=2.4, linewidth=1.2)
            secondary.set_ylabel(secondary_label, color=SECONDARY_COLOR)
            secondary.tick_params(axis="y", colors=SECONDARY_COLOR)
        else:
            # keep the panels visually aligned: no twin axis, same box
            pass

    OUT_GALLERY.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_GALLERY, dpi=300)
    plt.close(figure)
    print("wrote", OUT_GALLERY.relative_to(ROOT))

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _copy_with_retry(OUT_GALLERY, OUT_REPORT)
    print("copied", OUT_REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
