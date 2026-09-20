# PA-1: Beyond IID - Representation, Invariance and Rejection

Code for EE-5102/CS-6304 Programming Assignment 1 (Fall 2026).
Four tasks study what a robust representation should preserve, become
invariant to, and reject:

| Task | Topic | Status |
|------|--------------------------------------------|--------|
| 1 | Inductive biases: shape / texture / colour / spatial structure (STL-10; ResNet-50, ViT-B/16, CLIP) | planned |
| 2 | Unsupervised domain adaptation on PACS (Target = Sketch): Source-only, DAN, DANN, CDAN | implemented |
| 3 | Domain generalisation on PACS (Sketch unseen): ERM, DAN-DG, SAM | next |
| 4 | Open-set recognition (CIFAR-10 known vs CIFAR-100 unknowns): MSP/MLS/Energy/Mahalanobis, GCSC, PROSER | planned |

## Repository layout

```
common/          seeding, configs, logging, metrics, plotting
shared/          PACS dataset + frozen protocol, ResNet-18 backbone, MMD, GRL,
                 domain discriminator, frozen-BatchNorm policy
  splits/        committed split file (seed 6304) reused by Tasks 2 and 3
task2/           adaptation methods, single training pipeline, final evaluation
scripts/         fake-PACS generator + end-to-end CPU smoke test
docs/            short implementation / study notes
```

## Setup

```bash
conda create -n atml-pa1 python=3.11 -y
conda activate atml-pa1
pip install -r requirements.txt
```

## Data: PACS

PACS has 7 classes in 4 domains (photo, art_painting, cartoon, sketch). Either

```bash
# option A: download from the HuggingFace hub (needs internet + `datasets`)
python -m shared.prepare_pacs --from-hf --out data/pacs
```

or unpack a manual download so the four domain folders sit together and run

```bash
python -m shared.prepare_pacs --verify-only --out <folder>
```

Then create the frozen split file **once** and commit it:

```bash
python -m shared.build_splits --data-root data/pacs \
    --out shared/splits/pacs_sketch_seed6304.json
```

Tasks 2 and 3 must load this file (`--splits`), never regenerate it, so every
method sees exactly the same source train/val images.

## Task 2 - domain adaptation to Sketch

Four methods share one training pipeline; only the method-specific loss changes:

```bash
python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
```

Fixed across methods (see `task2/configs/base.yaml`): ResNet-18
`IMAGENET1K_V1` fully fine-tuned, AdamW 1e-4 / wd 1e-4, batch = 8 per source
domain (+24 target), <=30 epochs, early stop after 5 epochs without improvement
in mean source-validation macro-F1, seed 6304, BatchNorm running statistics
frozen at pretrained values.

Final evaluation (reads Sketch labels, so run only after all settings are
frozen):

```bash
python -m task2.evaluate_final --run-dir results/task2_dan_seed6304_... \
    --baseline-run results/task2_source_only_seed6304_...
python -m task2.evaluation.domain_separability --run-dir results/task2_dan_seed6304_...
```

Each run writes to `results/<run_id>/`:

```
config.json              merged config + hash
history.csv              one row per epoch (losses, per-domain val acc/F1)
metrics.json             best epoch, best mean source macro-F1
curves_train.png         loss curves (required evidence)
curves_val.png           source-validation curves
final_metrics.json       source-val + target metrics, per-class, confusions
target_outputs.npz       raw target logits/paths for failure analysis
domain_separability.json held-out accuracy of a source-vs-target probe
```

Checkpoints live in `checkpoints/<run_id>/{best,last}.pt` and are git-ignored.
`last.pt` supports `--resume` when a Kaggle session hits its time limit.

## Kaggle workflow (GPU)

1. Upload PACS once as a private dataset (or run `prepare_pacs --from-hf` in a
   notebook with internet enabled) and mount it at `/kaggle/input/...`.
2. In each notebook: `git clone <repo>`, `cd` into it, `pip install -r requirements.txt`.
3. Run **one method per notebook commit** so a failure costs one run, not all.
   Example:

```bash
python -m shared.build_splits --data-root /kaggle/input/pacs --out shared/splits/pacs_sketch_seed6304.json
python -m task2.train --config task2/configs/source_only.yaml --data-root /kaggle/input/pacs
```

4. Download `results/` into the repo, and publish `checkpoints/<erm_run_id>` as
   a private dataset so Task 3 can reuse the ERM baseline unchanged.
5. `--resume --run-id <same run id>` continues an interrupted run.

## Local smoke test (no GPU, no real data)

```bash
python scripts/run_smoke_test.py
```

This creates a fake PACS tree, builds splits, then checks:

* batch composition is exactly 8/8/8 source + 24 target;
* BatchNorm running statistics never change while gamma/beta still get gradients;
* all four methods train, checkpoint, evaluate and write their result files;
* `evaluate_final` and `domain_separability` run on a finished checkpoint.

## Reproducibility rules (assignment requirements)

* seed **6304** everywhere (`common/seed.py`);
* one committed split file reused by Tasks 2 and 3;
* frozen BatchNorm running statistics for every Task 2/3 method;
* checkpoint selection only on mean source-validation macro-F1;
* target labels are read only in `evaluate_final` / final analyses;
* every reported number must come from a saved JSON/CSV under `results/`.

## Attribution

External code is only used through public libraries (PyTorch / torchvision,
scikit-learn, matplotlib). Method implementations (MMD, gradient reversal,
CDAN conditioning) were written for this assignment from the papers listed in
the handout; add explicit notes here if any snippet is adapted later.
