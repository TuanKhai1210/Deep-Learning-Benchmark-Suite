# Notebooks

- `a1_colab_training.ipynb` orchestrates A1 main runs on Colab using the same
  `dlbench.a1.trainer.fit` implementation as the package CLI. It does not
  duplicate the training loop or evaluate the official test set.
- `a2_training.ipynb` starts from a local original TrashCan-Material dataset,
  reuses existing extracted data or its local archive and downloads only if
  both are missing, runs EDA fresh from that dataset on every session, renders
  that run's full label/split/pixel statistics and QA samples inline, checks
  real image/mask samples, and calls the shared A2 trainer for one independently
  configured U-Net or SegFormer run. Each EDA execution gets a unique local
  output folder; it does not mount Google Drive or reuse the preliminary EDA
  archive.
- The notebook checks GPU availability, the committed split and train-only
  normalization, then offers a synthetic GPU batch pilot and an optional
  real-data smoke run before its strict main-run gate. The pilot and smoke are
  not benchmark results. Main training requires all five configs to pass and
  writes run artifacts and a commit-pinned ledger to Google Drive.
- The CNN, RNN and Transformer candidate configs use batch size 64 and CUDA
  after a GPU batch pilot. Review and merge those choices before enabling main
  runs; the source-checkout cell also makes the editable `src/` package visible
  to the current notebook kernel.

Keep reusable model, data, training, evaluation and analysis logic in
`src/dlbench/`. The package CLI remains independently usable and testable;
the notebook is an orchestration interface, not the sole implementation.
Do not commit executed notebook outputs, raw data or checkpoints.
