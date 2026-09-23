"""Per-conflict model predictions for the Task 1 report.

Reads the cached conflict features (produced by ``task1.run_task1 --stage cache``)
and the trained linear heads, then writes one row per accepted cue conflict with
every classifier's prediction and whether that prediction is the content class
(shape decision), the style class (texture decision) or another class.

    # from the pulled Kaggle output
    python scripts/task1_conflict_examples.py \
        --cache-dir kaggle_outputs/t1_task1/PA_1/task1/cache \
        --heads-dir kaggle_outputs/t1_task1/PA_1/results/task1/heads \
        --manifest kaggle_outputs/t1_task1/PA_1/task1/data/cue_conflicts/manifest.json \
        --out results/task1/conflict_examples.csv

No model predictions are used to select or filter conflicts; this script only
reports what the frozen models did.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PREDICTORS = ["resnet50", "vit_b16", "clip_vit_b32", "clip_vit_b32_zero_shot"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="task1/cache")
    parser.add_argument("--heads-dir", default="results/task1/heads")
    parser.add_argument("--manifest", default="task1/data/cue_conflicts/manifest.json")
    parser.add_argument("--classes-file", default="task1/data/subsets_stl10_seed6304.json")
    parser.add_argument("--out", default="results/task1/conflict_examples.csv")
    parser.add_argument("--print-examples", type=int, default=3)
    return parser.parse_args(argv)


def head_predictions(head_path: Path, features: np.ndarray) -> Optional[np.ndarray]:
    if not head_path.exists():
        return None
    state = torch.load(head_path, map_location="cpu", weights_only=False)
    weight = state["state_dict"]["weight"].numpy()
    bias = state["state_dict"]["bias"].numpy()
    return (features @ weight.T + bias).argmax(axis=1)


def load_conflict_cache(cache_dir: Path, name: str):
    path = cache_dir / f"{name}_conflict.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data["files"], data["features"]


def main(argv=None) -> None:
    args = parse_args(argv)
    cache_dir, heads_dir = Path(args.cache_dir), Path(args.heads_dir)
    classes = json.loads(Path(args.classes_file).read_text(encoding="utf-8"))["classes"]
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_info: Dict[str, Tuple[str, str]] = {}
    for entry in manifest.get("pairs", {}).values():
        for item in (entry.get("items") or []):
            if "file" in item:
                manifest_info[item["file"]] = (entry["content_class"], entry["style_class"])

    predictions: Dict[str, Dict[str, int]] = {name: {} for name in PREDICTORS}
    file_order: List[str] = []
    content_of: Dict[str, str] = {}
    style_of: Dict[str, str] = {}

    for name in ["resnet50", "vit_b16", "clip_vit_b32"]:
        loaded = load_conflict_cache(cache_dir, name)
        if loaded is None:
            continue
        files, features = loaded
        preds = head_predictions(heads_dir / f"{name}.pt", features)
        if preds is None:
            continue
        for position, filename in enumerate(files):
            filename = str(filename)
            if filename not in file_order:
                file_order.append(filename)
                class_names = manifest_info.get(filename)
                if class_names is None:
                    raise SystemExit(f"{filename} missing from the manifest items list")
                content_of[filename], style_of[filename] = class_names
            predictions[name][filename] = int(preds[position])

    clip_cache = load_conflict_cache(cache_dir, "clip_vit_b32")
    zero_shot_path = cache_dir / "clip_zero_shot.npz"
    if clip_cache is not None and zero_shot_path.exists():
        files, features = clip_cache
        text = np.load(zero_shot_path)["text_features"]
        scale = float(np.load(zero_shot_path)["scale"])
        zero_preds = (scale * (features @ text.T)).argmax(axis=1)
        for position, filename in enumerate(files):
            predictions["clip_vit_b32_zero_shot"][str(filename)] = int(zero_preds[position])

    rows = []
    for filename in file_order:
        content_name, style_name = content_of[filename], style_of[filename]
        content_index, style_index = classes.index(content_name), classes.index(style_name)
        row = {"file": filename, "content_class": content_name, "style_class": style_name}
        shape_count = texture_count = 0
        for name in PREDICTORS:
            prediction = predictions[name].get(filename)
            if prediction is None:
                row[f"{name}_prediction"] = ""
                row[f"{name}_type"] = ""
                continue
            row[f"{name}_prediction"] = classes[prediction]
            if prediction == content_index:
                kind = "shape"
                shape_count += 1
            elif prediction == style_index:
                kind = "texture"
                texture_count += 1
            else:
                kind = "other"
            row[f"{name}_type"] = kind
        row["shape_decisions"] = shape_count
        row["texture_decisions"] = texture_count
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} conflicts)")

    # curated illustrative selections (agreement / disagreement / failure)
    def show(title: str, selection: List[Dict]) -> None:
        print(f"\n-- {title} --")
        for row in selection[: args.print_examples]:
            summary = ", ".join(
                f"{name}->{row[f'{name}_prediction']}({row[f'{name}_type']})" for name in PREDICTORS
            )
            print(f"  {row['file']}  content={row['content_class']} style={row['style_class']}  {summary}")

    show("agreements (all four predict shape)", [r for r in rows if r["shape_decisions"] == 4])
    show("disagreements (at least one texture decision)",
         [r for r in rows if r["texture_decisions"] >= 1])
    show("other-class failures (at least two 'other' decisions)",
         [r for r in rows if (4 - r["shape_decisions"] - r["texture_decisions"]) >= 2])
    texture_only = [r for r in rows if r["texture_decisions"] >= 2]
    show("texture-dominant cases (at least two texture decisions)", texture_only)


if __name__ == "__main__":
    main()
