# Evidence map: where every required item lives

Quick index for writing the report. Each item lists the saved artefact (all
committed under `results/`) and, where useful, the command that regenerates it.

## Task 1 - inductive biases (STL-10)

| Required evidence | Artefact |
|---|---|
| Clean / grayscale / extra-colour / patch-shuffle comparison | `results/task1/analysis.json` (`conditions`) and `results/report_tables/task1_conditions.csv` |
| Shape / texture / other counts + shape bias + coverage | `results/task1/analysis.json` (`conflicts`) and `results/report_tables/task1_conflicts.csv` |
| Translation curve | `results/task1/translation_curve.png` + `analysis.json` (`translation`) |
| Representation stability per intervention | `analysis.json` (`stability`, cosine clean-vs-transformed) - includes grayscale, hue, patch shuffle, all translations and **cue conflicts** (paired with their clean content image) |
| t-SNE / UMAP clean vs transformed | `results/task1/tsne_<backbone>_<condition>.png` |
| Informative conflict examples | `results/task1/conflict_examples.csv` (all 265 conflicts with every classifier's prediction and shape/texture/other type) + `report/figures/conflict_*.png` (6 example images) |

Regenerate: `python -m task1.run_task1 --config task1/configs/base.yaml --stage all`
(Kaggle kernel: `kaggle/kernels/t1_task1`).

## Task 2 - unsupervised domain adaptation (PACS -> Sketch)

| Required evidence | Artefact |
|---|---|
| Main table (source per-domain, mean, target, delta, separability) | `results/report_tables/task2_main_and_study.csv` |
| Training / alignment-loss curves | `results/t2_*/curves_train.png`, `curves_val.png`, `history.csv` |
| Per-class target changes + confusions | `results/t2_*/target_per_class.csv`, `target_confusions.csv` |
| Controlled study (lambda_MMD 0.1 / 1 / 10) | rows `t2_dan`, `t2_dan_lambda01`, `t2_dan_lambda10` in the table |
| Target outputs for failure cases | `results/t2_*/target_outputs.npz` |

## Task 3 - domain generalisation (Sketch unseen)

| Required evidence | Artefact |
|---|---|
| Main table (ERM / DAN-DG / SAM, per-source, mean, worst, Sketch, delta) | `results/report_tables/task3_main_and_study.csv` |
| Source-domain separability (chance 33.3%) | `results/t3_*/source_domain_separability.json`, ERM: `results/t3_erm/` |
| Sharpness proxy (Delta L at radius 0.05) | `results/t3_*/sharpness.json`, ERM: `results/t3_erm/sharpness.json` |
| Training curves incl. MMD | `results/t3_dan_dg/curves_train.png`, `curves_val.png` |
| Controlled study (rho 0.01 / 0.05 / 0.1) | rows `t3_sam`, `t3_sam_rho001`, `t3_sam_rho01` |
| Per-class Sketch changes + failures vs Task 2 | `results/t3_*/target_per_class.csv`, `target_confusions.csv` |

## Task 4 - open-set recognition (CIFAR-10 vs CIFAR-100)

| Required evidence | Artefact |
|---|---|
| MSP / MLS / Energy / Mahalanobis table on vanilla | `results/t4_osr/table_posthoc.csv` |
| Vanilla / GCSC / PROSER table (CSA + OSR, MLS + placeholder) | `results/t4_osr/table_models.csv` |
| Score-distribution / ROC figure | `results/t4_osr/roc_scores.png` |
| >=3 near + >=3 far accepted failures (class, prediction, score, threshold) | `results/t4_osr/failure_near.csv`, `failure_far.csv` |
| PROSER calibration bias + placeholder score | `evaluate_osr` in `results/t4_osr/osr_metrics.json` |

## Cross-cutting

| Item | Artefact |
|---|---|
| Frozen split file (Tasks 2/3) | `shared/splits/pacs_sketch_seed6304.json` |
| CIFAR-10 split (Task 4) | `task4/data/splits/cifar10_seed6304.json` |
| STL-10 subsets (Task 1) | `task1/data/subsets_stl10_seed6304.json` (generated on Kaggle) |
| Agent configs per run | `results/<run_id>/config.json` + `git_commit.txt` |
| Method/implementation notes | `docs/methods_notes.md` |
| Kaggle reproducibility | `kaggle/kernels/*`, `scripts/kaggle_queue.py`, `scripts/kaggle_push.py` |

Commands that rebuild the tables: `python scripts/build_report_tables.py`.
