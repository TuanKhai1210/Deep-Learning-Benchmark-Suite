# Reproduce the A2 report and experiment

The experiment compares scratch U-Net (U), pretrained SegFormer-B0 with training color jitter off (S0), and the same SegFormer with training color jitter on (S1), at run seeds 69420, 67 and 69. All runs use one TrashCan-Material split with seed 36.

## Verify the printed results on CPU

From this folder, run `python reproduction/verify_results.py` with Python 3.10 or newer. The script needs only the standard library. It checks the nine bundled test JSONs and completion markers, recomputes the overlap metrics from the saved confusion counts, recovers the best epoch from each validation history, and checks means, sample SDs and paired differences. The checkpoint and test membership identifiers are preserved in each result. Full run IDs, experiment IDs, checkpoint/test hashes and artifact keys are in `verified_results.json`.

| Evidence | Use |
|---|---|
| `evidence/runs/<seed>/<U,S0,S1>/effective_config.json` | Model, seed, initialization, data and training policy |
| `evidence/runs/<seed>/<U,S0,S1>/history.json` | Per-epoch train/validation metrics and checkpoint selection |
| `test_best.json`, `test_complete.json`, `test_per_class.csv`, `test_confusion.csv` in each run folder | Full test metrics, artifact hashes and class counts |
| `inference_benchmark.json` in available run folders | Raw timing samples. The report compares only the matched seed-69 profile |
| `evidence/per_class_summary.csv`, `evidence/background_errors_by_seed.csv` | Cross-seed class statistics and per-seed background omissions |
| `evidence/manifest.json`, `evidence/example_batch.json` | Fixed split, regenerated batch IDs and target hashes |

## Regenerate the data example

1. Obtain `dataset.zip` from the [University of Minnesota TrashCan dataset record](https://doi.org/10.13020/g1gx-y834). The verified archive has 553,029,970 bytes and SHA-256 `caf93d992b85343cfc500386c94ca9ffa6c63e2284e04127753a40db8f33ba4c`. Extract the material version, retaining both `instances_*_trashcan.json` files and the original image directory names.
2. Use Python with PyTorch and torchvision, NumPy, Pillow **12.3.0**, and matplotlib. The reported Kaggle/Colab numerical profile uses PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130 and NumPy 2.1.3. The batch figure is CPU preprocessing using the pinned converter and loader.
3. From this folder, run `python reproduction/make_example_batch.py /path/to/dataset/material_version --output example-output`. Check the four image IDs and equality of both target tensors against `evidence/example_batch.json`.

## Train, resume and evaluate

- **Fresh experiment.** Import `reproduction/notebooks/fresh_kaggle_nine_runs.ipynb` into Kaggle, enable GPU and Internet, attach the verified `dataset.zip` as an input if the built-in download is inaccessible, and keep the default three models and three seeds. Run via **Save & Run All**. Record the actual accelerator and dependency profile. Use the latest saved notebook output as input after a budget pause. This recipe produces new runs in one Kaggle profile and has locally tested setup logic. The reported runs follow the historical path below.
- **Historical path.** For seed 69420, `colab_u_69420.ipynb` runs the U-Net and `colab_s0_s1_69420.ipynb` runs the two SegFormer variants. The notebooks cache the archive in `MyDrive/dlbench_a2_cache/dataset.zip`. The six remaining slots used `remaining_kaggle_67_69.ipynb`, then `continue_colab_67_69.ipynb` with the archived Kaggle suite as input. For the exact 7 November partial suite, `recover_colab_u69.ipynb` validates the known U69 replay discrepancy before evaluating the same selected checkpoint. Import the appropriate notebook, choose Tesla T4 and Python 3.13 where required, and point its settings at the matching saved suite. The continuation suite root on Drive contains `import_receipt.json`, `seed_67/` and `seed_69/`. Use the same output root and a single writer across sessions. Resume at the last completed epoch boundary.
- **Evaluate an existing checkpoint.** Recover the trusted original run folder containing `best.pt`, `metadata.json`, `effective_config.json` and its source snapshot. The source below matches the saved source digest for the nine runs. Set `PYTHONPATH` to `reproduction/core/src`, then run the command below. Pass a new output path and match the saved numerical environment for a comparable replay. The notebook's normal test stage additionally writes the completion marker and CSVs.

```sh
python -m dlbench.a2.cli evaluate --data-root /path/to/material_version \
  --manifest evidence/manifest.json --run-dir /path/to/original/run \
  --split test --device cuda --workers 0 --output test_replay.json
```

In PowerShell, set `$env:PYTHONPATH = (Resolve-Path reproduction/core/src).Path` and enter the command on one line. Checkpoint and dataset ZIPs are separate from this report source. Exact recovery requires the original exports and their identities. Fresh training can differ across accelerator/library profiles even with fixed seeds.

## Build the PDF

Open `main.tex` as the root document in a LaTeX project, or run `tectonic main.tex` from this folder. All figure files and the report class used by `main.tex` are included.
