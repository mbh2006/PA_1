"""Download PACS and export it to the ImageFolder-style layout used here.

Two options:

1. HuggingFace hub (needs internet + ``datasets``)::

       python -m shared.prepare_pacs --from-hf --out data/pacs

   Downloads ``flwrlabs/pacs`` (9,990 images; columns: image, domain, label)
   and writes ``data/pacs/<domain>/<class>/<index>.jpg``.

2. Manual download: obtain the original PACS archive (from the authors or a
   mirror), unpack it so that the four domain folders end up together, then::

       python -m shared.prepare_pacs --verify-only --out <folder>

On Kaggle you usually do this once, then save the exported folder as a private
dataset so later notebooks can mount it read-only without re-downloading.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from shared.pacs import CLASSES, DOMAINS, find_pacs_root, list_domain_samples


def export_from_hf(out_root: Path, limit: int | None = None) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'datasets' package is required for --from-hf. Install it with "
            "`pip install datasets`, or download PACS manually and use --verify-only."
        ) from exc

    print("Downloading flwrlabs/pacs from the HuggingFace hub ...")
    dataset = load_dataset("flwrlabs/pacs", split="train")
    label_feature = dataset.features["label"]
    out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, example in enumerate(dataset):
        if limit is not None and written >= limit:
            break
        domain = example["domain"]
        class_name = label_feature.int2str(example["label"])
        if domain not in DOMAINS or class_name not in CLASSES:
            raise ValueError(f"Unexpected PACS entry: domain={domain!r} label={class_name!r}")
        folder = out_root / domain / class_name
        folder.mkdir(parents=True, exist_ok=True)
        image = example["image"]
        image.convert("RGB").save(folder / f"{index:05d}.jpg", quality=95)
        written += 1
        if written % 1000 == 0:
            print(f"  wrote {written} images")
    print(f"Done: {written} images under {out_root}")


def verify(root: Path) -> None:
    pacs_root = find_pacs_root(root)
    print(f"PACS root: {pacs_root}")
    total = 0
    for domain in DOMAINS:
        samples = list_domain_samples(pacs_root, domain)
        total += len(samples)
        print(f"  {domain:13s} {len(samples):5d} images")
    print(f"  {'TOTAL':13s} {total:5d} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare / verify the PACS dataset.")
    parser.add_argument("--out", default="data/pacs", help="Where the dataset lives (or will be written).")
    parser.add_argument("--from-hf", action="store_true", help="Download and export from the HuggingFace hub.")
    parser.add_argument("--verify-only", action="store_true", help="Only check the folder layout and count images.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (debugging).")
    args = parser.parse_args()

    out = Path(args.out)
    if args.from_hf:
        export_from_hf(out, limit=args.limit)
    verify(out)


if __name__ == "__main__":
    main()
