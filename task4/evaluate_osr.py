"""Task 4 common evaluation: post-hoc scores, model comparison, failure cases.

Run after every checkpoint and score definition is frozen (CIFAR-100 never
influences training, selection, score design or thresholds).

    python -m task4.evaluate_osr \
        --vanilla-run results/t4_vanilla --gcsc-run results/t4_gcsc \
        --proser-run results/t4_proser

Outputs in ``--out-dir`` (default ``results/t4_osr``):

* ``table_posthoc.csv``  - MSP / MLS / Energy / Mahalanobis on the frozen vanilla model
* ``table_models.csv``   - Vanilla / GCSC / PROSER with CSA and MLS (+ PROSER placeholder)
* ``roc_scores.png``     - three-panel ROC figure for MSP / MLS / Mahalanobis
* ``failure_near.csv``, ``failure_far.csv`` - accepted unknowns under the vanilla MLS threshold
* ``osr_metrics.json``   - machine-readable copy of everything above
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc, roc_curve  # noqa: E402

from common.logging import get_logger  # noqa: E402
from task4.data.cifar10 import CIFAR10_CLASSES  # noqa: E402
from task4.evaluation.failure_analysis import accepted_unknowns  # noqa: E402
from task4.evaluation.metrics import auroc, operating_point  # noqa: E402
from task4.evaluation.thresholds import threshold_at_percentile  # noqa: E402
from task4.scores.energy import energy  # noqa: E402
from task4.scores.mahalanobis import fit_mahalanobis, mahalanobis  # noqa: E402
from task4.scores.mls import mls  # noqa: E402
from task4.scores.msp import msp  # noqa: E402

POSTHOC = ["MSP", "MLS", "Energy", "Mahalanobis"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Task 4 OSR evaluation.")
    parser.add_argument("--vanilla-run", default="results/t4_vanilla")
    parser.add_argument("--gcsc-run", default="results/t4_gcsc")
    parser.add_argument("--proser-run", default="results/t4_proser")
    parser.add_argument("--cache-dir", default="task4/cache")
    parser.add_argument("--out-dir", default="results/t4_osr")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--top-failures", type=int, default=8)
    return parser.parse_args(argv)


def load_cache(cache_dir: Path, run_dir: str) -> dict:
    path = cache_dir / f"{Path(run_dir).name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"cache {path} missing - run `python -m task4.extract_outputs --run-dir {run_dir}` first")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def score_dict(cache: dict, means: np.ndarray, covariance: np.ndarray) -> dict:
    """Unknownness scores for val / test / near / far from one cached model."""
    num_known = int(cache["num_known"])

    def known(split: str) -> np.ndarray:
        return cache[f"logits_{split}"][:, :num_known]

    splits = ("val", "test", "near", "far")
    scores = {}
    for name, function in [("MSP", msp), ("MLS", mls), ("Energy", energy)]:
        scores[name] = {split: function(known(split)) for split in splits}
    scores["Mahalanobis"] = {
        split: mahalanobis(cache[f"features_{split}"], means, covariance) for split in splits
    }
    return scores


def main(argv=None) -> None:
    args = parse_args(argv)
    logger = get_logger()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vanilla = load_cache(cache_dir, args.vanilla_run)
    gcsc = load_cache(cache_dir, args.gcsc_run)
    proser = load_cache(cache_dir, args.proser_run)

    # ------------------------------------------------------------------ Vanilla post-hoc
    means, covariance = fit_mahalanobis(
        vanilla["features_train"], vanilla["labels_train"], int(vanilla["num_known"]))
    posthoc_scores = score_dict(vanilla, means, covariance)

    posthoc_rows = []
    for name in POSTHOC:
        values = posthoc_scores[name]
        row = {"score": name}
        row["auroc_near"] = auroc(values["test"], values["near"])
        row["auroc_far"] = auroc(values["test"], values["far"])
        row["auroc_all_unknowns"] = auroc(values["test"], np.concatenate([values["near"], values["far"]]))
        operating = operating_point(values["val"], values["test"], values["near"])
        row.update({
            "threshold": operating["threshold"],
            "known_test_acceptance": operating["known_test_acceptance"],
            "near_rejection": operating["unknown_rejection"],
            "near_fpr95": operating["fpr_at_95tpr"],
        })
        far_operating = operating_point(values["val"], values["test"], values["far"])
        row["far_rejection"] = far_operating["unknown_rejection"]
        row["far_fpr95"] = far_operating["fpr_at_95tpr"]
        posthoc_rows.append(row)

    # ------------------------------------------------------------------ model comparison
    model_rows = []
    for label, cache in [("Vanilla", vanilla), ("GCSC", gcsc), ("PROSER", proser)]:
        num_known = int(cache["num_known"])
        known_test = cache["logits_test"][:, :num_known]
        predictions = known_test.argmax(axis=1)
        csa = float((predictions == cache["labels_test"]).mean())

        values = {split: mls(cache[f"logits_{split}"][:, :num_known])
                  for split in ("val", "test", "near", "far")}
        row = {
            "model": label,
            "score": "MLS",
            "closed_set_accuracy": csa,
            "auroc_near": auroc(values["test"], values["near"]),
            "auroc_far": auroc(values["test"], values["far"]),
            "auroc_all_unknowns": auroc(values["test"], np.concatenate([values["near"], values["far"]])),
        }
        operating = operating_point(values["val"], values["test"], values["near"])
        row.update({
            "threshold": operating["threshold"],
            "known_test_acceptance": operating["known_test_acceptance"],
            "near_rejection": operating["unknown_rejection"],
            "near_fpr95": operating["fpr_at_95tpr"],
        })
        far_operating = operating_point(values["val"], values["test"], values["far"])
        row["far_rejection"] = far_operating["unknown_rejection"]
        row["far_fpr95"] = far_operating["fpr_at_95tpr"]
        model_rows.append(row)

        if label == "PROSER":
            num_known = int(cache["num_known"])
            margins = {}
            for split in ("val", "test", "near", "far"):
                logits = cache[f"logits_{split}"]
                margins[split] = logits[:, :num_known].max(axis=1) - logits[:, num_known:].max(axis=1)
            bias = float(np.percentile(margins["val"], 95))
            placeholder_values = {split: margins[split] - bias for split in margins}
            row_ph = {
                "model": "PROSER",
                "score": "Placeholder",
                "closed_set_accuracy": csa,
                "auroc_near": auroc(placeholder_values["test"], placeholder_values["near"]),
                "auroc_far": auroc(placeholder_values["test"], placeholder_values["far"]),
                "auroc_all_unknowns": auroc(
                    placeholder_values["test"],
                    np.concatenate([placeholder_values["near"], placeholder_values["far"]])),
            }
            operating = operating_point(placeholder_values["val"], placeholder_values["test"],
                                        placeholder_values["near"])
            row_ph.update({
                "threshold": operating["threshold"],
                "known_test_acceptance": operating["known_test_acceptance"],
                "near_rejection": operating["unknown_rejection"],
                "near_fpr95": operating["fpr_at_95tpr"],
            })
            far_operating = operating_point(placeholder_values["val"], placeholder_values["test"],
                                            placeholder_values["far"])
            row_ph["far_rejection"] = far_operating["unknown_rejection"]
            row_ph["far_fpr95"] = far_operating["fpr_at_95tpr"]
            model_rows.append(row_ph)

    # ------------------------------------------------------------------ failure analysis
    vanilla_mls = posthoc_scores["MLS"]
    tau = threshold_at_percentile(vanilla_mls["val"], 95.0)
    fine_names = None
    try:
        from torchvision.datasets import CIFAR100
        fine_names = CIFAR100(root=args.data_root, train=False, download=True).classes
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load CIFAR-100 class names (%s); using indices", exc)

    failures = {"near": [], "far": []}
    for group in ("near", "far"):
        labels = vanilla[f"labels_{group}"]
        names = fine_names if fine_names is not None else [str(i) for i in range(100)]
        failures[group] = accepted_unknowns(
            vanilla_mls[group], tau, labels, names,
            vanilla["logits_" + group][:, :int(vanilla["num_known"])],
            CIFAR10_CLASSES, top_k=args.top_failures)

    # ------------------------------------------------------------------ figure
    figure_path = out_dir / "roc_scores.png"
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    comparisons = [
        ("Known vs Near", "near"),
        ("Known vs Far", "far"),
        ("Known vs All unknowns", "all"),
    ]
    for axis, (title, which) in zip(axes, comparisons):
        for name in POSTHOC:
            values = posthoc_scores[name]
            unknown = (np.concatenate([values["near"], values["far"]]) if which == "all"
                       else values[which])
            labels = np.concatenate([np.zeros(len(values["test"])), np.ones(len(unknown))])
            roc = np.concatenate([values["test"], unknown])
            fpr, tpr, _ = roc_curve(labels, roc)
            axis.plot(fpr, tpr, label=f"{name} (AUROC {auc(fpr, tpr):.3f})")
        axis.plot([0, 1], [0, 1], "k--", alpha=0.3)
        axis.set_title(title)
        axis.set_xlabel("FPR (unknown accepted)")
        axis.set_ylabel("TPR (unknown rejected)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)

    # ------------------------------------------------------------------ save
    output = {
        "posthoc_vanilla": posthoc_rows,
        "models": model_rows,
        "failure_near": failures["near"],
        "failure_far": failures["far"],
        "vanilla_mls_threshold": tau,
        "cache_dir": str(cache_dir),
        "runs": {"vanilla": args.vanilla_run, "gcsc": args.gcsc_run, "proser": args.proser_run},
    }
    with open(out_dir / "osr_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)

    for name, rows in [("table_posthoc.csv", posthoc_rows), ("table_models.csv", model_rows),
                       ("failure_near.csv", failures["near"]), ("failure_far.csv", failures["far"])]:
        if not rows:
            continue
        with open(out_dir / name, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Wrote %s (tables, failure CSVs, ROC figure)", out_dir)
    for row in posthoc_rows:
        logger.info("vanilla %-12s AUROC near %.4f far %.4f | near rej %.4f far rej %.4f",
                    row["score"], row["auroc_near"], row["auroc_far"],
                    row["near_rejection"], row["far_rejection"])
    for row in model_rows:
        logger.info("%-8s %-11s CSA %.4f | AUROC near %.4f far %.4f | near rej %.4f far rej %.4f",
                    row["model"], row["score"], row["closed_set_accuracy"],
                    row["auroc_near"], row["auroc_far"], row["near_rejection"], row["far_rejection"])


if __name__ == "__main__":
    main()
