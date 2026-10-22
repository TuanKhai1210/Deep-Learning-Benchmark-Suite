# Notebooks

- `a1_colab_training.ipynb` orchestrates A1 main runs on Colab using the same
  `dlbench.a1.trainer.fit` implementation as the package CLI. It does not
  duplicate the training loop or evaluate the official test set.
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

