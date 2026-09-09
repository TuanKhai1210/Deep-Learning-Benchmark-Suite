# AI Usage Log

## 2026-08-29 - Project planning and repository design

- Tool/model: OpenAI Codex 5.6 Sol
- Used by: Tạ Tuấn Khải
- Stage: Project planning
- Purpose: Analyze the handbook and plan repository/GitHub Pages structure
- Affected sections: Repository structure, project plan, GitHub Pages structure
- Prompt summary: Requested help interpreting the assignment and planning work
- AI contribution: Summarized requirements and proposed the initial structure
- Human verification: Compared the suggestions with the official handbook
- Responsible member: Tạ Tuấn Khải
- Verification sources: Course handbook and official GitHub documentation

## 2026-09-04 - Experiment planning and repository integration

- Tool/model: OpenAI Codex; exact model identifier for this interaction not recorded
- Used by: Tạ Tuấn Khải
- Stage: Assignment 1 planning and development setup
- Purpose: Define shared interfaces, prepare Python experiment configuration, and reconcile project documentation with the existing repository
- Affected sections: Python package setup, configurations, shared interfaces, command-line entry point, tests, developer documentation, project plan, README and GitHub Pages
- Prompt summary: Requested an implementation-ready folder structure, Python-only experiment configs, seed values, updated responsibilities for Thiên/Khoa/Khải, and repairs preserving earlier team/course information.
- AI contribution: Proposed file organization and contracts; added development utilities and pending implementation entry points; updated role/seed records and documentation; retained draft status for decisions not yet approved
- Human verification: Pending team review of the diff, commands, configuration, and assignment alignment; automated development checks do not establish correctness of unimplemented ML functions
- Automated verification : Python syntax checks passed; 31 tests passed and 5 ML tests remained explicitly skipped. Editable package installation and installed-package checks passed in a separate temporary directory using Python 3.12.14. GitHub-hosted CI and ML training were not run.
- Responsible member: Tạ Tuấn Khải; module owners Nguyễn Hạo Thiên and Nguyễn Anh Khoa review their assigned interfaces and requirements
- Verification sources: Existing repository history and member/course details, user-confirmed roles and seeds, and the course handbook retained for requirement checks
- Experimental status: No dataset preparation, model training, or benchmark measurements are claimed from this assistance

## 2026-09-06 - State saving requirements for training resumption

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 project initial implementaion
- Purpose: Technical clarification for training resumption state requirements
- Affected files/sections:
  - `src/dlbench/a1/trainer.py`
  - `src/dlbench/a1/checkpoint.py`
- Prompt summary: Inquired why optimizer, scheduler, and RNG states must be persisted alongside model weights for training resumption, and where state restoration should be executed.
- AI contribution: Explained the functional difference between `best.pt` (inference/evaluation artifact) and `last.pt` (execution resumption artifact), detailing why unpersisted optimizer momentum and RNG states disrupt training dynamics, and recommended decoupling state loading from state application to avoid side effects in I/O modules.
- Student verification: Reviewed PyTorch state persistence mechanics (`optimizer.state_dict()` and `torch.get_rng_state()`), designed optional empty payload support (`{}`) for runs without learning rate schedulers to pass schema validation safely.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification: Official [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training) on saving and loading models across general checkpoints.

## 2026-09-06 - Test generation for metrics and checkpointing logic

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 project initial implementaion
- Purpose: Test generation and validation framework design for shared classification metrics, parameter counting, and atomic checkpoint management.
- Affected files/sections:
  - `src/dlbench/a1/metrics.py`
  - `src/dlbench/a1/checkpoint.py`
  - `tests/test_metrics.py`
  - `tests/test_checkpoint.py`
- Prompt summary: Requested test suites for classification metrics and checkpoint functions.
- AI contribution: Provided unit test templates covering edge cases (`zero_division=0`, non-finite metric handling, tie-breaking hierarchies, atomic temporary file replacement).
- Student verification: Verified mathematical formulations for hand-calculated Macro-F1 scores, verified required dictionary keys against `contracts.py`, and confirmed all test cases pass cleanly with `python -m unittest discover -s tests -v` on Python.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification:
  - Official documentation for `torch.save`, `torch.load`, `sklearn.metrics.accuracy_score`, `sklearn.metrics.f1_score`and `torch.numel`.
  - Course Project Handbook CO3133 (Semester-261).

## 2026-09-06 - Steps to train a DL model

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 Training logic implementation
- Purpose: Clarification of canonical deep learning training step sequences inside an epoch loop. Affected files/sections:
- `src/dlbench/a1/engine.py` (`train_one_epoch`)
- Prompt summary: Inquired about the standard step sequence required to train a deep learning model within a single epoch loop.
- AI contribution: Outlined the canonical PyTorch execution order: setting `model.train()`, transferring tensors to device, clearing gradients (`optimizer.zero_grad()`), executing forward pass to compute logits, calculating loss with unnormalized logits, running backpropagation (`loss.backward()`), updating weights (`optimizer.step()`), and accumulating sample-weighted loss.
- Student verification: Implemented the step sequence inside `train_one_epoch`, confirmed logits are passed directly to `CrossEntropyLoss` without softmax, and verified via unit tests that parameter tensors mutate post-update.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification:
  - Official [PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html) on model training step lifecycle.

## 2026-09-07 - MLP test creation and coverage expansion

- Tool/model: GitHub Copilot; exact model identifier for this interaction not recorded
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 MLP implementation
- Purpose: Create and expand unit tests for the MLP classifier, including output dimensions, flattening behavior, dropout, parameter counts, supported activations, training updates, and missing or invalid parameters.
- Affected files/sections:
  - `src/dlbench/a1/models/mlp.py`
  - `src/dlbench/a1/metrics.py`
  - `tests/test_mlp.py`
- Prompt summary: Requested additional tests for the MLP, with emphasis on dimensions, dropout logic, parameter counts, and missing parameters.
- AI contribution: Added focused `unittest` cases for image and flattened inputs, exact linear-layer dimensions, dropout behavior in training and evaluation modes, parameter-count consistency, supported activations, gradient updates, and constructor validation.
- Student verification: Reviewed the tests against the current MLP implementation and ran `venv/Scripts/python.exe -m unittest -v tests.test_mlp`; all tests passed.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification: Existing MLP implementation, parameter-count helper, repository test conventions, and PyTorch module behavior.


## 2026-09-06 - Exploratory Data Analysis requirements and leakage constraints

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Hạo Thiên
- Stage: Assignment 1 data exploration and protocol design
- Purpose: Technical clarification of EDA scope for Fashion-MNIST and guidelines for data leakage prevention.
- Affected files/sections:
  - `src/dlbench/a1/data/eda.py`
  - Report Part 1 (Problem and Data Description, EDA narrative)
- Prompt summary: Inquired what EDA means within the assignment context, what specific checks are required for Fashion-MNIST, and why it must be performed before model training.
- AI contribution: Clarified required analytical components (class balance, shape/dtype verification, pixel range bounds, and representative grid visualizations) and emphasized that normalization statistics must be computed strictly on the training partition to prevent leakage.
- Student verification: Audited Fashion-MNIST class labels against Zalando's official specifications, confirmed all 10 classes are evenly represented, and reviewed the handbook Section 3.1 and 13 data requirements.
- Responsible member: Nguyễn Hạo Thiên
- Sources used for verification: Course Project Handbook CO3133 (Semester-261) Section 3.1, 10, and 13.

## 2026-09-07 - Stratified dataset splitting and manifest persistence

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Hạo Thiên
- Stage: Assignment 1 data pipeline implementation
- Purpose: Design and verification of deterministic stratified splitting logic adhering to the `SplitManifest` contract.
- Affected files/sections:
  - `src/dlbench/a1/data/split.py`
  - `configs/a1/splits/fashion_mnist_split.json`
- Prompt summary: Requested implementation checking for `create_split`, `validate_split`, `save_split`, and `load_split` using a fixed split seed.
- AI contribution: Finished class-proportional stratified partitioning over official training indices using `numpy.random.default_rng(split_seed=36)`, added assertions to verify index disjointness and full partition coverage, and structured atomic JSON serialization that refuses silent overwriting of existing split files.
- Student verification: Executed `split.py` via `test_pipeline.py`, confirmed the generation of `configs/a1/splits/fashion_mnist_split.json`, verified that official test indices ($0\dots9{,}999$) are strictly isolated in a separate namespace, and validated index counts against `SplitManifest` attributes in `contracts.py`.
- Responsible member: Nguyễn Hạo Thiên
- Sources used for verification:
  - `dlbench/a1/contracts.py` (`SplitManifest` definition)
  - Official NumPy documentation on `Generator.shuffle` and random sampling reproducibility.

## 2026-09-07 - DataLoader construction and batch contract implementation

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Hạo Thiên
- Stage: Assignment 1 data pipeline implementation and verification
- Purpose: Build reproducible PyTorch DataLoaders adhering to the shared Batch and DataLoaders contract.
- Affected files/sections:
  - `src/dlbench/a1/data/loaders.py`
  - `test_pipeline.py`
- Prompt summary: Requested implementation check for `collate_samples` and `build_dataloaders`.
- AI contribution: Successfully fixed `collate_samples` to stack images and labels while preserving sample IDs, set up `build_dataloaders` with a dedicated run-seeded `torch.Generator` and `seed_worker` for training, configured deterministic evaluation loaders without shuffling, added safe smoke subset handling.
- Student verification: Created `test_pipeline.py` to check the code. Verified `loaders.py` imports and interfaces against `contracts.py`; executed `test_pipeline.py` to confirm batch collation outputs match expected dimensions (`[16, 1, 28, 28]` float32 tensors, `int64` labels, and aligned `sample_ids` string lists) for train, validation, and test loaders.
- Responsible member: Nguyễn Hạo Thiên
- Sources used for verification:
  - `dlbench/a1/contracts.py` (`Batch`, `DataLoaders`)
  - PyTorch documentation for `torch.utils.data.DataLoader` (worker initialization, generators, and collation mechanics)

## 2026-09-07 - Stratified dataset splitting and manifest persistence

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Hạo Thiên
- Stage: Assignment 1 data pipeline implementation
- Purpose: Design and verification of deterministic stratified splitting logic adhering to the `SplitManifest` contract.
- Affected files/sections:
  - `src/dlbench/a1/data/split.py`
  - `configs/a1/splits/fashion_mnist_split.json`
- Prompt summary: Requested implementation checking for `create_split`, `validate_split`, `save_split`, and `load_split` using a fixed split seed.
- AI contribution: Finished class-proportional stratified partitioning over official training indices using `numpy.random.default_rng(split_seed=36)`, added assertions to verify index disjointness and full partition coverage, and structured atomic JSON serialization that refuses silent overwriting of existing split files.
- Student verification: Executed `split.py` via `test_pipeline.py`, confirmed the generation of `configs/a1/splits/fashion_mnist_split.json`, verified that official test indices ($0\dots9{,}999$) are strictly isolated in a separate namespace, and validated index counts against `SplitManifest` attributes in `contracts.py`.
- Responsible member: Nguyễn Hạo Thiên
- Sources used for verification:
  - `dlbench/a1/contracts.py` (`SplitManifest` definition)
  - Official NumPy documentation on `Generator.shuffle` and random sampling reproducibility.

## 2026-09-07 - DataLoader construction and batch contract implementation

- Tool: Gemini 3.8 Flash
- Used by: Nguyễn Hạo Thiên
- Stage: Assignment 1 data pipeline implementation and verification
- Purpose: Build reproducible PyTorch DataLoaders adhering to the shared Batch and DataLoaders contract.
- Affected files/sections:
  - `src/dlbench/a1/data/loaders.py`
  - `test_pipeline.py`
- Prompt summary: Requested implementation check for `collate_samples` and `build_dataloaders`.
- AI contribution: Successfully fixed `collate_samples` to stack images and labels while preserving sample IDs, set up `build_dataloaders` with a dedicated run-seeded `torch.Generator` and `seed_worker` for training, configured deterministic evaluation loaders without shuffling, added safe smoke subset handling.
- Student verification: Created `test_pipeline.py` to check the code. Verified `loaders.py` imports and interfaces against `contracts.py`; executed `test_pipeline.py` to confirm batch collation outputs match expected dimensions (`[16, 1, 28, 28]` float32 tensors, `int64` labels, and aligned `sample_ids` string lists) for train, validation, and test loaders.
- Responsible member: Nguyễn Hạo Thiên
- Sources used for verification:
  - `dlbench/a1/contracts.py` (`Batch`, `DataLoaders`)
  - PyTorch documentation for `torch.utils.data.DataLoader` (worker initialization, generators, and collation mechanics)
