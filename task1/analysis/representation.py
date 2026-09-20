"""Representation-level analysis: cosine stability and t-SNE visualisation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def cosine_stability(clean_features: np.ndarray, transformed_features: np.ndarray) -> float:
    """Mean cosine similarity between paired clean and transformed features."""
    a = np.asarray(clean_features, dtype=np.float64)
    b = np.asarray(transformed_features, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return float((a * b).sum(axis=1).mean())


def tsne_projection(clean_features: np.ndarray, transformed_features: np.ndarray,
                    seed: int = 6304, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    combined = np.vstack([clean_features, transformed_features]).astype(np.float64)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(perplexity, len(combined) // 4 or 1),
                init="pca", learning_rate="auto")
    return tsne.fit_transform(combined)


def tsne_figure(clean_features: np.ndarray, transformed_features: np.ndarray,
                labels: np.ndarray, class_names, out_path: str | Path,
                title: str = "", seed: int = 6304, perplexity: float = 30.0,
                marker_label: str = "transformed") -> Path:
    """One 2-D projection fit on combined clean + transformed features.

    Colour encodes the ground-truth class; marker style distinguishes clean
    (circles) from transformed (crosses) examples. Report the settings in the
    text: t-SNE with perplexity and seed as configured here.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embedding = tsne_projection(clean_features, transformed_features, seed, perplexity)
    n = len(clean_features)

    plt.figure(figsize=(8, 6))
    colour_map = plt.get_cmap("tab10")
    for label in sorted(set(int(v) for v in labels)):
        mask_clean = np.asarray(labels) == label
        mask_transformed = mask_clean
        plt.scatter(embedding[:n][mask_clean, 0], embedding[:n][mask_clean, 1],
                    s=12, color=colour_map(label % 10), alpha=0.75,
                    label=str(class_names[label]) if True else None)
        plt.scatter(embedding[n:][mask_transformed, 0], embedding[n:][mask_transformed, 1],
                    s=18, color=colour_map(label % 10), alpha=0.75, marker="x")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color="grey", label="clean"),
               plt.Line2D([], [], marker="x", linestyle="", color="grey", label=marker_label)]
    legend_one = plt.legend(handles=handles, loc="upper right", fontsize=8, title="condition")
    plt.gca().add_artist(legend_one)
    plt.legend(fontsize=6, loc="lower right", ncols=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
