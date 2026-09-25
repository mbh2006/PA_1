"""Compact Task 2 training-dynamics figure (one panel per adaptation method).

    DAN : classification loss + MMD
    DANN: classification loss + domain-discriminator accuracy
    CDAN: classification loss + domain-discriminator accuracy

Solid curves use the left axis (classification loss), dashed curves the right
axis (alignment signal). A grey dotted line marks chance (0.5) for the
discriminator panels. Values come straight from the run histories
(``results/t2_*/history.csv``), no recomputation.

    python scripts/make_task2_training_curves.py

Writes ``doc_image/task2/task2_training_curves.png`` and
``report/figures/task2_training_curves.png``.
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
RUNS = [("t2_dan", "DAN", "MMD"),
        ("t2_dann", "DANN", "domain accuracy"),
        ("t2_cdan", "CDAN", "domain accuracy")]
OUT_GALLERY = ROOT / "doc_image" / "task2" / "task2_training_curves.png"
OUT_REPORT = ROOT / "report" / "figures" / "task2_training_curves.png"

CLS_COLOR = "tab:blue"
SECONDARY_COLOR = "tab:orange"


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

    return {
        "epochs": [int(row["epoch"]) for row in rows],
        "cls": column("train_cls_loss"),
        "mmd": column("train_mmd"),
        "dom_acc": column("train_dom_acc"),
    }


def main() -> None:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8.5,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "figure.dpi": 200})

    figure = plt.figure(figsize=(7.0, 2.7))
    grid = figure.add_gridspec(1, 3, left=0.055, right=0.945, top=0.845,
                               bottom=0.175, wspace=0.55)

    for column, (run, title, secondary_label) in enumerate(RUNS):
        history = load_history(run)
        primary = figure.add_subplot(grid[0, column])
        secondary = primary.twinx()

        primary.plot(history["epochs"], history["cls"], color=CLS_COLOR,
                     marker="o", markersize=2.6, linewidth=1.2)
        series = history["mmd"] if run == "t2_dan" else history["dom_acc"]
        secondary.plot(history["epochs"][:len(series)], series,
                       color=SECONDARY_COLOR, linestyle="--", marker="s",
                       markersize=2.4, linewidth=1.2)

        primary.set_title(title)
        primary.set_xlabel("epoch")
        primary.set_xticks(history["epochs"])
        primary.set_xticklabels([str(value) if value % 2 == 1 or value == history["epochs"][-1]
                                 else "" for value in history["epochs"]])
        primary.grid(axis="y", alpha=0.25)
        primary.tick_params(axis="y", colors=CLS_COLOR)
        secondary.tick_params(axis="y", colors=SECONDARY_COLOR)
        if column == 0:
            primary.set_ylabel("classification loss", color=CLS_COLOR)
        secondary.set_ylabel(secondary_label, color=SECONDARY_COLOR)

        if run != "t2_dan":
            secondary.axhline(0.5, color="grey", linestyle=":", linewidth=0.9)
            secondary.text(history["epochs"][0], 0.503, "chance", color="grey",
                           fontsize=6, va="bottom")
            secondary.set_ylim(0.38, 0.90)

    OUT_GALLERY.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_GALLERY, dpi=300)
    plt.close(figure)
    print("wrote", OUT_GALLERY.relative_to(ROOT))

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    _copy_with_retry(OUT_GALLERY, OUT_REPORT)
    print("copied", OUT_REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
