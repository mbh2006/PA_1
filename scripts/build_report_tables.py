"""Assemble the report tables from saved result files.

    python scripts/build_report_tables.py

Reads ``results/`` (Task 2/3 runs, Task 4 OSR evaluation) and
``results/task1/analysis.json`` and writes machine-readable tables plus a
Markdown summary under ``results/report_tables/``. Every number traces back to
a saved JSON, as the assignment requires.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

RESULTS = Path("results")
OUT = RESULTS / "report_tables"

TASK2_RUNS = ["t2_source_only", "t2_dan", "t2_dann", "t2_cdan"]
TASK2_STUDY = ["t2_dan", "t2_dan_lambda01", "t2_dan_lambda10"]
TASK3_RUNS = [("ERM", "t2_source_only"), ("DAN-DG", "t3_dan_dg"), ("SAM", "t3_sam")]
TASK3_STUDY = ["t3_sam", "t3_sam_rho001", "t3_sam_rho01"]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(name: str, rows: List[Dict]) -> None:
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(OUT / name, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT / name}")


def task2_table() -> List[Dict]:
    rows = []
    for run in TASK2_RUNS + [r for r in TASK2_STUDY if r not in TASK2_RUNS]:
        metrics = load_json(RESULTS / run / "final_metrics.json")
        if not metrics:
            continue
        separability = load_json(RESULTS / run / "domain_separability.json") or {}
        source = metrics["source_validation"]
        row = {
            "run": run,
            "method": metrics["method"],
            "source_mean_accuracy": sum(v["accuracy"] for v in source.values()) / len(source),
            "source_mean_macro_f1": sum(v["macro_f1"] for v in source.values()) / len(source),
            "target_accuracy": metrics["target"]["accuracy"],
            "target_macro_f1": metrics["target"]["macro_f1"],
            "domain_separability": separability.get("score"),
        }
        for domain, values in source.items():
            row[f"source_{domain}_accuracy"] = values["accuracy"]
            row[f"source_{domain}_macro_f1"] = values["macro_f1"]
        if "target_accuracy_change_vs_baseline" in metrics:
            row["target_accuracy_change"] = metrics["target_accuracy_change_vs_baseline"]
            row["target_macro_f1_change"] = metrics["target_macro_f1_change_vs_baseline"]
        rows.append(row)
    return rows


def task3_table() -> List[Dict]:
    rows = []
    for label, run in TASK3_RUNS:
        metrics = load_json(RESULTS / run / "final_metrics.json")
        if not metrics:
            continue
        out_dir = RESULTS / ("t3_erm" if label == "ERM" else run)
        separability = load_json(out_dir / "source_domain_separability.json") or {}
        sharpness = load_json(out_dir / "sharpness.json") or {}
        row = {
            "method": label,
            "run": run,
            "source_mean_accuracy": metrics.get("source_mean_accuracy"),
            "source_mean_macro_f1": metrics.get("source_mean_macro_f1"),
            "source_worst_macro_f1": metrics.get("source_worst_macro_f1"),
            "target_accuracy": metrics["target"]["accuracy"],
            "target_macro_f1": metrics["target"]["macro_f1"],
            "source_domain_separability": separability.get("score"),
            "sharpness_delta": sharpness.get("sharpness_delta"),
        }
        if "target_accuracy_change_vs_erm" in metrics:
            row["target_accuracy_change"] = metrics["target_accuracy_change_vs_erm"]
            row["target_macro_f1_change"] = metrics["target_macro_f1_change_vs_erm"]
        for domain, values in metrics["source_validation"].items():
            row[f"source_{domain}_macro_f1"] = values["macro_f1"]
        rows.append(row)

    for run in TASK3_STUDY:
        if any(row["run"] == run for row in rows):
            continue
        metrics = load_json(RESULTS / run / "final_metrics.json")
        if not metrics:
            continue
        separability = load_json(RESULTS / run / "source_domain_separability.json") or {}
        sharpness = load_json(RESULTS / run / "sharpness.json") or {}
        rows.append({
            "method": f"{run} (study)",
            "run": run,
            "source_mean_accuracy": metrics.get("source_mean_accuracy"),
            "source_mean_macro_f1": metrics.get("source_mean_macro_f1"),
            "source_worst_macro_f1": metrics.get("source_worst_macro_f1"),
            "target_accuracy": metrics["target"]["accuracy"],
            "target_macro_f1": metrics["target"]["macro_f1"],
            "source_domain_separability": separability.get("score"),
            "sharpness_delta": sharpness.get("sharpness_delta"),
        })
    return rows


def task4_tables() -> None:
    metrics = load_json(RESULTS / "t4_osr" / "osr_metrics.json")
    if not metrics:
        return
    write_csv("task4_posthoc_vanilla.csv", metrics["posthoc_vanilla"])
    write_csv("task4_models.csv", metrics["models"])


def task1_table() -> List[Dict]:
    analysis = load_json(RESULTS / "task1" / "analysis.json")
    if not analysis:
        return []
    rows = []
    for condition, per_backbone in analysis.get("conditions", {}).items():
        for backbone, values in per_backbone.items():
            rows.append({
                "condition": condition,
                "backbone": backbone,
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                "mean_max_confidence": values["mean_max_confidence"],
                "prediction_consistency": values.get("consistency"),
                "cosine_stability": analysis.get("stability", {}).get(condition, {}).get(
                    backbone.replace("_zero_shot", ""), {}).get("cosine_clean_vs_transformed"),
            })
    return rows


def task1_conflicts() -> List[Dict]:
    analysis = load_json(RESULTS / "task1" / "analysis.json")
    if not analysis:
        return []
    rows = []
    for backbone, predictors in analysis.get("conflicts", {}).items():
        for predictor, values in predictors.items():
            rows.append({"backbone": backbone, "predictor": predictor,
                         **{k: values.get(k) for k in ("shape", "texture", "other", "total",
                                                       "shape_bias", "coverage")}})
    return rows


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    task2_rows = task2_table()
    task3_rows = task3_table()
    task1_rows = task1_table()
    conflict_rows = task1_conflicts()
    write_csv("task2_main_and_study.csv", task2_rows)
    write_csv("task3_main_and_study.csv", task3_rows)
    write_csv("task1_conditions.csv", task1_rows)
    write_csv("task1_conflicts.csv", conflict_rows)
    task4_tables()

    lines = ["# Result tables (auto-generated)\n"]
    sections = [("Task 1 - clean / interventions (per backbone)", task1_rows),
                ("Task 1 - cue conflicts (shape bias + coverage)", conflict_rows),
                ("Task 2 - main comparison and lambda_MMD study", task2_rows),
                ("Task 3 - main comparison and rho study", task3_rows)]
    task4_metrics = load_json(RESULTS / "t4_osr" / "osr_metrics.json")
    if task4_metrics:
        sections.append(("Task 4 - post-hoc scores on the vanilla model", task4_metrics["posthoc_vanilla"]))
        sections.append(("Task 4 - model comparison (MLS + PROSER placeholder)", task4_metrics["models"]))
    for title, rows in sections:
        if not rows:
            continue
        lines.append(f"\n## {title}\n")
        lines.append(markdown_table(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
