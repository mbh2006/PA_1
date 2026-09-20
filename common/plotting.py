"""Small plotting helpers (Agg backend so they work headless on Kaggle)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(
    history: List[Dict[str, float]],
    y_keys: Iterable[str],
    out_path: str | Path,
    title: str = "",
    x_key: str = "epoch",
) -> Path:
    """Plot one or more columns of a run's history.csv."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for key in y_keys:
        xs = [row[x_key] for row in history if key in row]
        ys = [row[key] for row in history if key in row]
        if xs:
            plt.plot(xs, ys, marker="o", ms=3, label=key)
    plt.xlabel(x_key)
    plt.ylabel("value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
