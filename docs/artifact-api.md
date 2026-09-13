# Artifact API for trainer integration

Import these helpers from `dlbench.common.artifacts`.

- `save_run_metadata(run_dir, config)` writes the run configuration, environment,
  source snapshots and metadata, returning the saved metadata.
- `checkpoint_provenance(metadata)` maps that metadata to checkpoint fields.
- `compute_data_provenance(config)` returns `split_hash` and `statistics_hash`
  without writing files or requiring `run.mode`, `_sources`, Git or environment
  capture. Both metadata creation and this helper share the same hash builders.
  File hashes cover raw bytes; statistics use sorted, compact JSON.

Before evaluation or resume, compare both returned hashes with the checkpoint.
The helper does not validate split membership, model configuration, seed or
scheduler compatibility; those checks remain the caller's responsibility.

## History

`append_history(run_dir, row)` uses the existing nine-column trainer schema:
`epoch`, `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`,
`train_macro_f1`, `val_macro_f1`, `learning_rate`, `epoch_seconds`.

Epochs are nonnegative integers and must increase strictly. Scores are in [0, 1];
loss, learning rate and elapsed seconds must be finite and nonnegative.
Missing/extra fields, corrupt existing rows and repeated epochs are rejected
before writing. One process owns each run; concurrent writers are unsupported.
If history extends beyond a resumed checkpoint, reconcile it explicitly before
training. This API never silently truncates history.

## Metrics

`save_metrics(run_dir, metrics)` requires `eval_split`, `epoch`, `timing_scope`,
`timing_units` (seconds), and loss/accuracy/macro-F1 fields. Validation uses
`val_*` keys and `metrics.json`; test uses `test_*` keys and `metrics_test.json`.
Additional JSON-serializable provenance fields are preserved. The writer does
not infer a checkpoint, fabricate durations or supply missing scores.

Existing metric files are refused, including on resume. A trainer that supports
replacing final results must first agree on a versioning/replacement policy.
For now the caller must arrange a fresh result destination. Run identity and
hashes remain in the adjacent metadata.json; callers should supply checkpoint
identity explicitly when exporting metrics separately from their run directory.

## Verification

`python -m unittest discover -s tests -v`

Local verification on 2026-09-13: 97 tests run, OK, 3 existing skips.
