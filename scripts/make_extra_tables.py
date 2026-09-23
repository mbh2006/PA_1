"""Supplementary per-example tables for the Task 2/3/4 report packs.

Tasks 2 & 3 (from each run's ``target_outputs.npz``):
* the most confident mistakes of every method,
* examples the method *fixed* relative to the shared ERM baseline,
* examples the method *broke* relative to ERM.

Task 4 (from the cached vanilla outputs):
* quantiles of every novelty score for known / near-known / far-unknown inputs.

    python scripts/make_extra_tables.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


def load_outputs(path: Path):
    data = np.load(path, allow_pickle=True)
    return (data["labels"], data["logits"], [str(p) for p in data["paths"]])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path.relative_to(ROOT))


def examples_for(task: str, baseline_run: str, method_runs: Sequence[Tuple[str, str]],
                 class_names: Sequence[str], out_path: Path) -> None:
    baseline_labels, baseline_logits, baseline_paths = load_outputs(
        RESULTS / baseline_run / "target_outputs.npz")
    baseline_predictions = baseline_logits.argmax(axis=1)
    baseline_map = {path: (int(label), int(pred)) for path, label, pred
                    in zip(baseline_paths, baseline_labels, baseline_predictions)}

    rows: List[Dict[str, object]] = []
    for label, run in method_runs:
        labels, logits, paths = load_outputs(RESULTS / run / "target_outputs.npz")
        predictions = logits.argmax(axis=1)
        confidence = softmax(logits).max(axis=1)

        wrong = np.where(predictions != labels)[0]
        for index in wrong[np.argsort(-confidence[wrong])][:5]:
            rows.append({
                "task": task, "method": label, "kind": "confident_error",
                "path": paths[index], "true_class": class_names[labels[index]],
                "method_prediction": class_names[predictions[index]],
                "erm_prediction": class_names[baseline_map[paths[index]][1]]
                if paths[index] in baseline_map else "",
                "confidence": round(float(confidence[index]), 4),
            })

        fixed = [i for i, path in enumerate(paths)
                 if path in baseline_map and baseline_map[path][1] != baseline_map[path][0]
                 and predictions[i] == labels[i]]
        broken = [i for i, path in enumerate(paths)
                  if path in baseline_map and baseline_map[path][1] == baseline_map[path][0]
                  and predictions[i] != labels[i]]
        for kind, selection in [("fixed_vs_erm", fixed), ("broken_vs_erm", broken)]:
            for index in sorted(selection, key=lambda i: -confidence[i])[:5]:
                rows.append({
                    "task": task, "method": label, "kind": kind,
                    "path": paths[index], "true_class": class_names[labels[index]],
                    "method_prediction": class_names[predictions[index]],
                    "erm_prediction": class_names[baseline_map[paths[index]][1]],
                    "confidence": round(float(confidence[index]), 4),
                })
    write_csv(out_path, rows)
    for row in rows:
        print("  ", row["method"], row["kind"], row["true_class"], "->",
              row["method_prediction"], f"(erm {row['erm_prediction']})", row["confidence"])


def score_summary(cache_path: Path, out_path: Path) -> None:
    from task4.scores.energy import energy
    from task4.scores.mahalanobis import fit_mahalanobis, mahalanobis
    from task4.scores.mls import mls
    from task4.scores.msp import msp

    with np.load(cache_path, allow_pickle=True) as data:
        cache = {key: data[key] for key in data.files}
    known = int(cache["num_known"])
    means, covariance = fit_mahalanobis(cache["features_train"], cache["labels_train"], known)
    score_functions = {
        "MSP": lambda split: msp(cache[f"logits_{split}"][:, :known]),
        "MLS": lambda split: mls(cache[f"logits_{split}"][:, :known]),
        "Energy": lambda split: energy(cache[f"logits_{split}"][:, :known]),
        "Mahalanobis": lambda split: mahalanobis(cache[f"features_{split}"], means, covariance),
    }
    rows = []
    for name, function in score_functions.items():
        for split in ("test", "near", "far"):
            values = function(split)
            rows.append({
                "score": name,
                "split": {"test": "known_test", "near": "near_unknown", "far": "far_unknown"}[split],
                "n": int(len(values)),
                "mean": round(float(values.mean()), 4),
                "p05": round(float(np.percentile(values, 5)), 4),
                "median": round(float(np.median(values)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
            })
    write_csv(out_path, rows)
    for row in rows:
        print("  ", row)


def main() -> None:
    from shared.pacs import CLASSES

    print("== Task 2 examples ==")
    examples_for("task2", "t2_source_only",
                 [("Source-only", "t2_source_only"), ("DAN", "t2_dan"),
                  ("DANN", "t2_dann"), ("CDAN", "t2_cdan")],
                 CLASSES, RESULTS / "t2_target_examples.csv")

    print("== Task 3 examples ==")
    examples_for("task3", "t2_source_only",
                 [("DAN-DG", "t3_dan_dg"), ("SAM", "t3_sam")],
                 CLASSES, RESULTS / "t3_target_examples.csv")

    print("== Task 4 score summary ==")
    cache = ROOT / "kaggle_outputs" / "t4_rest" / "PA_1" / "task4" / "cache" / "t4_vanilla.npz"
    if cache.exists():
        score_summary(cache, RESULTS / "t4_osr" / "score_summary.csv")
    else:
        print("skip score summary: cache missing", cache)


if __name__ == "__main__":
    main()
