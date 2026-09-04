# Generated runs — not committed

Each experiment gets one unique directory, e.g.:
`a1/a1_fashion_mnist_cnn_v1_seed36_run01/`.

Run each main model with seeds `36, 69420, 67, 69` on the same saved split.

Required artifacts:

- Original model/protocol Python CONFIG files and resolved config.json.
- environment.json and provenance (commit, dirty state, split hash, statistics).
- history.csv, metrics.json, predictions.csv.
- best.pt; last.pt if resumability is implemented.
- Generated figures.

The trainer and artifact writer are under development. Smoke runs must be marked
and stored separately from benchmark runs. Copy only reviewed small tables/figures
to `reports/a1/` and `docs/assets/a1/`.
