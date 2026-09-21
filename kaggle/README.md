# Kaggle kernels (API-driven runs)

Every reported run was executed as a **Kaggle script kernel** pushed from this
machine with the Kaggle API. Each kernel clones this public repository at the
current commit, stages the attached datasets, runs the same pipeline as the
local commands in the top-level README, and writes its results into
`/kaggle/working` for a one-file download.

Generated folders (do not edit by hand - regenerate with `make_kernels.py`):

| Kernel slug | Task | Run id | Attached datasets |
|---|---|---|---|
| `pa1-t2-source-only` | 2 | `t2_source_only` | PACS |
| `pa1-t2-dan` | 2 | `t2_dan` | PACS, ERM results + checkpoint |
| `pa1-t2-dann` | 2 | `t2_dann` | PACS, ERM results + checkpoint |
| `pa1-t2-cdan` | 2 | `t2_cdan` | PACS, ERM results + checkpoint |
| `pa1-t2-dan-lambda01` | 2 (study) | `t2_dan_lambda01` | PACS, ERM results + checkpoint |
| `pa1-t2-dan-lambda10` | 2 (study) | `t2_dan_lambda10` | PACS, ERM results + checkpoint |
| `pa1-t3-dan-dg` | 3 | `t3_dan_dg` | PACS, ERM results |
| `pa1-t3-sam` | 3 | `t3_sam` | PACS, ERM results |
| `pa1-t3-sam-rho001` | 3 (study) | `t3_sam_rho001` | PACS, ERM results |
| `pa1-t3-sam-rho01` | 3 (study) | `t3_sam_rho01` | PACS, ERM results |
| `pa1-t3-erm-eval` | 3 diagnostics | `t3_erm` | PACS, ERM results + checkpoint (ran locally instead, see `results/t3_erm/NOTE.md`) |
| `pa1-t4-vanilla` | 4 | `t4_vanilla` | CIFAR-10, CIFAR-100 |
| `pa1-t4-gcsc-proser` | 4 | `t4_gcsc`, `t4_proser` | CIFAR-10, CIFAR-100, vanilla ckpt + cache |
| `pa1-t1` | 1 | `task1` | STL-10 binary files |

Datasets used (private unless public): `mbh2006/pacs-dataset01`,
`mbh2006/pa1-t2-erm-results`, `mbh2006/pa1-t2-erm-ckpt`,
`mbh2006/pa1-t4-vanilla-ckpt`, `mbh2006/pa1-t4-vanilla-cache`,
`pankrzysiu/cifar10-python`, `robsonricardodasilva/cifar-100-python`,
`pratt3000/stl10-binary-files`.

The Task 2 ERM results/checkpoint are the shared baseline for Task 3 and are
reused unchanged (never retrained).

## Helper scripts

```powershell
python kaggle/make_kernels.py                       # regenerate all kernel folders

python scripts/kaggle_push.py push  <name>          # start a headless GPU commit
python scripts/kaggle_push.py watch <name>          # poll until it finishes
python scripts/kaggle_push.py pull  <name>          # download the output

python scripts/kaggle_queue.py <names...>           # queue many, max 2 GPU sessions
python scripts/kaggle_autopull.py <names...>        # unattended wait + pull
```

## Operational notes (learned the hard way, kept for reproducibility)

* Dataset mounts can appear either as `/kaggle/input/<slug>` or as
  `/kaggle/input/datasets/<owner>/<slug>`; all kernels discover both layouts
  with glob patterns.
* Install OpenCLIP with `--no-deps` plus its pure-python dependencies
  (`ftfy regex timm safetensors huggingface_hub`); letting pip resolve
  dependencies can pull a fresh torch/CUDA stack and kill the session.
* The Task 1 kernel mirrors every stage and any traceback into
  `STATUS.txt` in its output, and the training pipeline's own log into
  `pipeline.log`, so failures are diagnosable even when Kaggle's log stream is
  empty.
* Task 1 reads STL-10 directly from the read-only dataset mount instead of
  copying 2.6 GB into the working disk.
