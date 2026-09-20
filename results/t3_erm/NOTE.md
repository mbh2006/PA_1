# Task 3 ERM baseline diagnostics

The ERM rows of the Task 3 table come from the Task 2 source-only checkpoint,
reused unchanged, exactly as the assignment requires.

* Sketch metrics: produced by the original Task 2 evaluation
  (`results/t2_source_only/final_metrics.json`).
* Source-domain separability and the sharpness proxy: computed with the same
  scripts (`task3.evaluation.source_domain_separability`, `task3.evaluation.sharpness`)
  on CPU because the Kaggle evaluation-only kernel (`pa1-t3-erm-eval`) hit a
  transient dataset-mount error. The command used:

```powershell
python -m task3.evaluation.source_domain_separability --model-run results/t2_source_only \
    --out-dir results/t3_erm --data-root <PACS root> --device cpu
python -m task3.evaluation.sharpness --model-run results/t2_source_only \
    --out-dir results/t3_erm --data-root <PACS root> --device cpu
```

Both scripts are deterministic given the checkpoint and the fixed seed-6304
batches, so the numbers are identical to what the Kaggle kernel would produce.
