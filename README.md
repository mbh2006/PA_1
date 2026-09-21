# PA-1: Beyond IID - Representation, Invariance and Rejection

Code and results for **EE-5102/CS-6304 Programming Assignment 1 (Fall 2026)**.
All four tasks are complete and every reported number traces back to a saved
result file under `results/`.

| Task | Question | Status |
|------|----------|--------|
| 1 | Inductive biases on STL-10: shape / texture / colour / spatial structure (ResNet-50, ViT-B/16, CLIP) | complete |
| 2 | Unsupervised domain adaptation on PACS -> Sketch: Source-only, DAN, DANN, CDAN | complete |
| 3 | Domain generalisation with Sketch unseen: ERM (reused from T2), DAN-DG, SAM | complete |
| 4 | Open-set recognition: CIFAR-10 known vs CIFAR-100 unknowns; MSP/MLS/Energy/Mahalanobis, Vanilla/GCSC/PROSER | complete |

Report-side entry points: `results/report_tables/summary.md` (all required
tables, auto-assembled), `docs/evidence_map.md` (requirement -> artefact),
`docs/methods_notes.md` (method and design-decision notes, including the
frozen-BatchNorm stability findings).

## Results at a glance

**Task 1 - accuracy on the 500-image balanced test subset and cue-conflict summary**
(265 accepted conflicts, 395 rejected by the model-free rule; "conflict stability" is
the cosine similarity between each accepted conflict and its clean content image).

| Model | Clean | Grayscale | Hue rot. | Patch shuffle | Shape bias | Coverage | Conflict stability |
|---|---|---|---|---|---|---|---|
| ResNet-50 (V2) | 0.982 | 0.968 | 0.940 | 0.900 | 87.3% | 86.4% | 0.571 |
| ViT-B/16 | 0.984 | 0.960 | 0.968 | 0.912 | 98.0% | 94.3% | 0.587 |
| CLIP linear head | 0.976 | 0.954 | 0.956 | 0.798 | 94.5% | 89.8% | 0.798 |
| CLIP zero-shot | 0.948 | 0.928 | 0.930 | 0.782 | 92.1% | 90.2% | - |

**Task 2 - PACS -> Sketch**

| Method | Source mean macro-F1 | Sketch accuracy | Sketch macro-F1 | Domain separability |
|---|---|---|---|---|
| Source-only | 0.9375 | 0.6747 | 0.6578 | 0.9986 |
| DAN | 0.9320 | 0.7753 | 0.7382 | 0.8948 |
| DANN | 0.9357 | 0.6320 | 0.6431 | 1.0000 |
| CDAN | 0.9268 | 0.3156 | 0.4316 | 0.9959 |
| DAN lambda=0.1 (study) | 0.9231 | 0.7376 | 0.7533 | 0.9918 |
| DAN lambda=10 (study) | 0.0639 | collapsed | - | 0.8702 |

**Task 3 - PACS with Sketch unseen**

| Method | Mean source macro-F1 | Worst source macro-F1 | Sketch accuracy | Sketch macro-F1 | Source separability | Sharpness delta |
|---|---|---|---|---|---|---|
| ERM (T2 checkpoint) | 0.9375 | - | 0.6747 | 0.6578 | 0.8618 | 0.273 |
| DAN-DG | 0.9123 | 0.8926 | 0.7137 | 0.7048 | 0.6776 | 134.2 |
| SAM (rho=0.05) | 0.9583 | 0.9372 | 0.7137 | 0.7301 | 0.8553 | 0.166 |
| SAM rho=0.01 (study) | 0.9458 | 0.9305 | 0.6747 | 0.7098 | 0.852 | 0.216 |
| SAM rho=0.1 (study) | 0.9389 | 0.9173 | 0.7035 | 0.7111 | 0.816 | 0.118 |

**Task 4 - open-set recognition** (threshold = 95th percentile of unknownness on
CIFAR-10 validation; unknown rejection at that operating point)

| Model | Score | Closed-set acc. | AUROC near | AUROC far | Near rejection | Far rejection |
|---|---|---|---|---|---|---|
| Vanilla | MLS | 0.9446 | 0.787 | 0.884 | 0.300 | 0.531 |
| GCSC | MLS | 0.9503 | 0.818 | 0.912 | 0.356 | 0.578 |
| PROSER | MLS | 0.9408 | 0.764 | 0.854 | 0.260 | 0.466 |
| PROSER | Placeholder | 0.9408 | 0.754 | 0.872 | 0.263 | 0.500 |

Post-hoc scores on the frozen vanilla model: MSP is strongest for near unknowns
(AUROC 0.809), Mahalanobis for far unknowns (AUROC 0.916).

## Repository layout

```
common/          seeding (6304), config inheritance + hash, run logging, metrics, plots
shared/          PACS dataset + frozen protocol, ResNet-18 backbone, MMD, GRL,
                 domain discriminator, frozen-BatchNorm policy, PACS/CIFAR prep
task1/           STL-10 subsets, interventions (grayscale/hue/translation/shuffle),
                 AdaIN cue conflicts, ResNet-50/ViT-B16/CLIP backbones, analysis
task2/           Task 2 methods (source_only/dan/dann/cdan) + one training pipeline
                 + final evaluation + domain separability
task3/           Task 3 methods (erm/dan_dg/sam) + Sketch evaluation + source-domain
                 separability + sharpness proxy
task4/           CIFAR ResNet-18, Vanilla/GCSC/PROSER, novelty scores, output cache,
                 OSR evaluation and failure analysis
kaggle/          generated Kaggle script kernels for every reported run + generator
scripts/         smoke tests, fake-data generators, Kaggle push/queue/autopull,
                 report-table builder
docs/            methods notes and the evidence map
results/         every run's config/history/metrics + analysis artefacts
report/figures/  example cue-conflict images used in the report
```

## Setup

```bash
conda create -n atml-pa1 python=3.11 -y
conda activate atml-pa1
pip install -r requirements.txt
```

On Kaggle almost everything is preinstalled; the only extras needed are

```bash
pip install -q pyyaml scikit-learn tqdm
# Task 1 only, and NOT with dependencies (validated recipe):
pip install -q --no-deps open_clip_torch
pip install -q --no-deps ftfy regex timm safetensors huggingface_hub
```

## Data

* **PACS** (Tasks 2-3): `python -m shared.prepare_pacs --from-hf --out data/pacs`
  or unzip a manual download and run `--verify-only`. Then freeze the split once:
  `python -m shared.build_splits --data-root data/pacs --out shared/splits/pacs_sketch_seed6304.json`
  (committed, reused by both tasks).
* **CIFAR-10 / CIFAR-100** (Task 4): torchvision downloads them into `data/`;
  the CIFAR-10 split is committed at `task4/data/splits/cifar10_seed6304.json`,
  and the near/far CIFAR-100 groups are fixed in `task4/data/cifar100_unknowns.py`.
* **STL-10** (Task 1): torchvision loads the official binary files; the frozen
  subsets (80/20 split + balanced 500 test IDs) are committed at
  `task1/data/subsets_stl10_seed6304.json`.

## Reproducing each task

```bash
# Task 1 - all stages (prepare, conflicts, cache, heads, analysis)
python -m task1.run_task1 --config task1/configs/base.yaml --stage all

# Task 2
python -m task2.train --config task2/configs/source_only.yaml
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
python -m task2.evaluate_final --run-dir results/t2_dan --baseline-run results/t2_source_only
python -m task2.evaluation.domain_separability --run-dir results/t2_dan

# Task 3 (ERM baseline = the Task 2 source-only checkpoint, reused unchanged)
python -m task3.train --config task3/configs/dan_dg.yaml
python -m task3.train --config task3/configs/sam.yaml
python -m task3.evaluate_sketch --model-run results/t3_sam --erm-run results/t2_source_only
python -m task3.evaluation.source_domain_separability --model-run results/t3_sam
python -m task3.evaluation.sharpness --model-run results/t3_sam

# Task 4
python -m task4.train --config task4/configs/vanilla.yaml
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train --config task4/configs/proser.yaml --init-ckpt checkpoints/t4_vanilla/best.pt
python -m task4.extract_outputs --run-dir results/t4_vanilla     # + gcsc, proser
python -m task4.evaluate_osr --vanilla-run results/t4_vanilla \
    --gcsc-run results/t4_gcsc --proser-run results/t4_proser

# assemble the report tables from the saved result files
python scripts/build_report_tables.py
```

## Kaggle execution (how the reported runs were produced)

`kaggle/make_kernels.py` generates one script kernel per run; they clone this
repository, stage the datasets and execute the same pipeline as above.

```powershell
python kaggle/make_kernels.py                       # regenerate kernel folders
python scripts/kaggle_push.py push  t2_source_only  # start a headless GPU run
python scripts/kaggle_push.py watch t2_source_only
python scripts/kaggle_push.py pull  t2_source_only  # download outputs
python scripts/kaggle_queue.py <names...>           # queue respecting the 2-GPU-session limit
```

Notes for reproducing on Kaggle: attach the datasets (PACS export, CIFAR-10/100
mirrors, STL-10 binary files) as inputs; dataset mounts may appear either under
`/kaggle/input/<slug>` or `/kaggle/input/datasets/<owner>/<slug>` and the
kernels discover both layouts.

## Reproducibility and evidence discipline

* seed **6304** for every split, initialisation and comparison;
* committed frozen splits reused across methods and tasks
  (`shared/splits/pacs_sketch_seed6304.json`, `task4/data/splits/cifar10_seed6304.json`,
  `task1/data/subsets_stl10_seed6304.json`);
* BatchNorm running statistics frozen at pretrained values for all Task 2/3
  methods; checkpoints selected only on mean source-validation macro-F1;
* target labels read only in the final evaluation scripts; Task 3 never loads
  Sketch during training or selection; Task 4 thresholds calibrated on known
  validation data only;
* alignment objectives consume L2-normalised features (see
  `docs/methods_notes.md` for the frozen-BN scale-instability evidence);
* every reported number comes from `results/<run_id>/`; run configurations and
  commit hashes are stored with each result.

## Attribution

Method implementations were written for this assignment from the referenced
papers (DAN/MMD, DANN/GRL, CDAN, SAM, PROSER, AdaIN cue conflicts). Public
libraries used: PyTorch/torchvision, OpenCLIP, scikit-learn, matplotlib, PyYAML.
Generated artefacts (datasets, checkpoints, caches, Kaggle working copies) are
excluded from git via `.gitignore`.
