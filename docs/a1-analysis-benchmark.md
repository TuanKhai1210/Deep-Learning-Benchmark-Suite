# Saved-run analysis and inference benchmark

These tools do not train, select a new checkpoint, or evaluate the official test set.
Review this shared implementation with owners A/B before merging.

## Learning curves, without a GPU or dataset

```sh
python -m dlbench.a1.cli analyze --run-dir "runs/a1/YOUR_RUN"
```

This creates `analysis/learning_curves.png` and `analysis/summary.json`.
The summary retains the run ID, seed, code state, split/statistics hashes and input
artifact hashes. It checks saved best metrics against contiguous history and verifies
source snapshot hashes. It refuses an existing output directory. For a separate
destination use `analyze_run(run_dir, output_dir=destination)` in Python.

Curves display epochs starting at **1**; checkpoint/history epochs start at **0**.
Training metrics include augmentation/dropout effects; validation metrics use evaluation
mode. A lower validation loss than training loss alone is not evidence of leakage.
The sum of `epoch_seconds` is not inference latency or the full fit wall-clock time.

## Colab: validation errors and GPU benchmark

Use the team's trusted checkpoints only. Run from the repository root after installing
the project with `python -m pip install -e ".[ml]"`. Use the same model implementation
as the saved run, with this analysis change applied; do not silently rebuild from a
new tuned config. The script reads the run's **saved `config.json`**.

1. Select a T4 runtime, as used in the supplied runs.
2. Mount Drive containing the complete run directories, including `best.pt`, histories,
   metadata and source snapshots.
3. Restore the exact saved split manifest and dataset location. Merely using the same
   split seed does not establish a matching file hash. Checkpoint evaluation rejects
   mismatched split/statistics hashes; do not bypass this validation.
4. Execute for each selected run, choosing a new output directory:

```sh
python scripts/postprocess_run.py \
  --run-dir "/content/drive/MyDrive/dlbench_runs/a1/YOUR_RUN" \
  --output-dir "/content/drive/MyDrive/dlbench_analysis/YOUR_RUN" \
  --trust-checkpoint --benchmark
```

Adjust Drive paths to the actual mount. Omit `--benchmark` for validation analysis only.
No automatic device fallback is used for benchmark measurements. Runtime/data access
must already be configured; evaluation may use the dataset's existing download policy.

Outputs:

- Learning curves and provenance summary.
- `validation/predictions_validation.csv`: original sample IDs, targets and probabilities.
- `validation/confusion_validation.png`: all ten labels, including absent labels.
- `validation/errors.json`: confusion counts and misclassified sample IDs sorted by
  confidence; use these IDs to inspect source images. This is not yet an image montage.
- `benchmark.json`: device/environment, checkpoint hash, raw timing samples, median
  batch ms, amortized ms/image, throughput and parameter count.

Re-evaluation checks agreement with saved validation metrics before writing analysis.
GPU benchmark settings come from saved `timing`: FP32, batch 64, 20 warm-up forwards,
100 measured forwards for the agreed runs. It uses preallocated zero tensors of the
configured input shape for both models. Timing excludes data loading, host-to-device
transfer and checkpoint loading. CUDA is synchronized around measurements. TF32 and
matmul precision settings are recorded; use identical settings for compared models.
Amortized ms/image at batch 64 is **not** batch-one request latency.

Use the same runtime/GPU and settings for both models, avoid competing GPU work, and
retain raw measurements. CPU tests only verify implementation behavior, not T4 performance.

## Reporting

Keep baseline and adjusted runs separate. Report the saved best validation checkpoint,
not the last epoch. Disclose dirty runs and changed budgets/schedulers/dropout. Current
single-seed results cannot support a mean/std across the planned three seeds.
`compare_runs` produces per-run validation rows after compatibility checks; it does not
pool unlike configurations or rank hardware timing. Review matched seed sets and full
training budgets before any later aggregation. Do not choose hyperparameters on test data.
