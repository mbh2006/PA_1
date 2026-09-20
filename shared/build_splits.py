"""Create / verify the frozen PACS split file used by Tasks 2 and 3.

Usage (once, ideally on Kaggle right after PACS is mounted)::

    python -m shared.build_splits --data-root data/pacs \
        --out shared/splits/pacs_sketch_seed6304.json

Then commit the JSON. Both tasks must load the file, not regenerate it, so the
exact same images are used across every method and every session.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.seed import SEED
from shared.pacs import DOMAINS, find_pacs_root
from shared.pacs_protocol import build_splits, load_splits, save_splits, split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the PACS split file.")
    parser.add_argument("--data-root", required=True, help="Folder containing photo/..., sketch/...")
    parser.add_argument("--out", default="shared/splits/pacs_sketch_seed6304.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--verify", action="store_true",
                        help="Recompute and check equality with an existing file.")
    args = parser.parse_args()

    root = find_pacs_root(args.data_root)
    print(f"PACS root: {root}")

    if args.verify:
        existing = load_splits(args.out)
        rebuilt = build_splits(root, seed=args.seed, train_frac=args.train_frac)
        same = existing["domains"] == rebuilt["domains"]
        print("VERIFY:", "identical" if same else "MISMATCH")
        if not same:
            raise SystemExit(1)
        splits = existing
    else:
        splits = build_splits(root, seed=args.seed, train_frac=args.train_frac)
        save_splits(splits, args.out)
        print(f"Wrote {args.out}")

    summary = split_summary(splits)
    for domain in DOMAINS:
        parts = ", ".join(f"{k}: {v}" for k, v in summary[domain].items())
        print(f"  {domain:13s} {parts}")


if __name__ == "__main__":
    main()
