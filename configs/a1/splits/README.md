# Versioned split artifacts — owner A, reviewer B

This directory is intentionally empty of actual data indices until preparation
runs. Commit the real split JSON and small provenance/normalization metadata.
Do not put these files under ignored `data/`.

The primary manifest is `fashion_mnist_seed36.json`, created with the agreed
split seed `36`.
Its fields follow `src/dlbench/a1/contracts.py:SplitManifest`, plus a schema
version, class counts, dataset provenance and hash recorded by the implementation.

Only train and validation indices share the official-training namespace.
Official test indices reference a separate source. Preserve sample IDs such as
`fashion_mnist:official_train:123` versus `fashion_mnist:official_test:123`.

Never generate replacement indices on each run. Changed split = new protocol
version and rerun affected comparisons. mean/std must come only from unaugmented
training samples. Generate the actual split/statistics files during data preparation.
