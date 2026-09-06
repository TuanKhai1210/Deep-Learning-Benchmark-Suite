# Tests

From the repository root, install the package and run:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

- `test_config.py`: configuration and CLI checks; no torch/data required.
- `test_model_contract.py`: model shape/backward tests, pending model implementation.
- `test_pipeline_contract.py`: metric, checkpoint and split contracts, pending implementation.
- `test_metrics.py`: isolated classification and model size checking.
- `test_checkpoint.py`: checkpoint saving and loading.

Tests for unfinished ML functionality are explicitly skipped. Enable them as each module is completed, and add DataLoader determinism, train-only statistics, checkpoint reload, tiny-overfit and end-to-end integration tests.

CI currently checks configuration and Python code without downloading datasets or training models. Passing these checks does not establish model correctness. When enabling ML tests, install compatible ML dependencies in the test environment and update CI accordingly. Existing passing tests must not be skipped to hide failures.
