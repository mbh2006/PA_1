# Kaggle kernels (API-driven runs)

These folders are **Kaggle script kernels** pushed with the Kaggle API from the
project root using ``scripts/kaggle_push.py``. Each kernel:

1. clones the public GitHub repo at the current commit,
2. discovers the attached PACS dataset under ``/kaggle/input``,
3. runs one Task-2 method (train -> final evaluation -> domain separability),
4. writes ``results_<run_id>.zip`` for a one-file download.

Generated folders (do not edit by hand, regenerate with ``make_kernels.py``):

| Kernel slug            | Method       | Attaches                                        |
|------------------------|--------------|-------------------------------------------------|
| `pa1-t2-source-only`   | source_only  | `pa1-pacs`                                      |
| `pa1-t2-dan`           | dan          | `pa1-pacs`, `pa1-t2-erm` (baseline + checkpoint)|
| `pa1-t2-dann`          | dann         | `pa1-pacs`, `pa1-t2-erm`                        |
| `pa1-t2-cdan`          | cdan         | `pa1-pacs`, `pa1-t2-erm`                        |

The ERM checkpoint is published once as the private dataset `pa1-t2-erm`; it is
the shared baseline for Task 3 and must be reused unchanged.

### Local commands

```powershell
# regenerate kernel folders
python kaggle/make_kernels.py

# push (starts a headless GPU commit), poll, download outputs
python scripts/kaggle_push.py push  t2_source_only
python scripts/kaggle_push.py watch t2_source_only
python scripts/kaggle_push.py pull  t2_source_only
```
