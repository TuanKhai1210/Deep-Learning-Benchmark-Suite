# Generated runs — not committed

Each experiment gets one unique directory, e.g.:
`a1/a1_fashion_mnist_cnn_v1_seed69420_run01/`.

Create the data split once with seed `36`. Run each main model with training
seeds `69420`, `67` and `69` on that same saved split.

Required artifacts:

- Original model/protocol Python CONFIG files and resolved config.json.
- environment.json and provenance (commit, dirty state, split hash, statistics).
- history.csv, metrics.json, predictions.csv.
- best.pt; last.pt if resumability is implemented.
- Generated figures.

The trainer and artifact writer are under development. Smoke runs must be marked
and stored separately from benchmark runs. Copy only reviewed small tables/figures
to `reports/a1/` and `docs/assets/a1/`.
