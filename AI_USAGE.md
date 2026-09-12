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

## 2026-09-08 - Trainer test generation and training-loop coverage

- Tool/model: GitHub Copilot; OpenAI ChatGPT 5.6 Luna
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 trainer implementation
- Purpose: Generate tests for trainer calculations, actual loss reduction, checkpoint evaluation, early stopping, and periodic checkpoint saving.
- Affected files/sections:
  - `src/dlbench/a1/trainer.py`
  - `tests/test_trainer.py`
- Prompt summary: Requested trainer tests that verify calculations using real training loss, evaluate checkpoint behavior, and early stopping/checkpoint frequency logic.
- AI contribution: Added tests for real two-epoch loss reduction using a deterministic model and DataLoader, validation/test checkpoint loader selection, checkpoint configuration and split-hash mismatch rejection, optimizer settings, validation metric calculations, early stopping monitors, patience, and save frequency.
- Student verification: Reviewed the tests against the trainer and checkpoint contracts and ran `python -m unittest -v tests.test_trainer`.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification: Existing trainer, engine, checkpoint, contract, and artifact interfaces; PyTorch optimizer and loss behavior.

## 2026-09-08 - GitHub Pages visual redesign

- Tool/model: OpenAI Codex; exact model identifier not recorded
- Used by / responsible member: Tạ Tuấn Khải
- Purpose: Redesign the research portfolio and assignment pages while preserving confirmed team roles, split seed 36 and run seeds 69420, 67, 69
- Affected sections: docs site configuration, layouts, styles, interactive conceptual architecture diagrams, assignment pages and experiment-contract presentation
- AI contribution: Authored site design and frontend code; reorganized existing content; corrected two stale four-seed references in the contract to three run seeds
- Human verification: Pending team review of content, accessibility, course alignment and the deployed GitHub Pages build
- Verification scope: Local preview checks are documented with the delivered update; they do not verify the hosted Jekyll build or ML implementations
- Experimental status: No model training, evaluation or measured benchmark results were generated or claimed

### Readability revision requested on 2026-09-08

- User feedback: Enlarge small text, make interactive controls obvious, brighten backgrounds and emphasize important figures and labels.
- Changes: Stronger typography and contrast; visible link/button treatments; highlighted counts and seeds; clearer member cards; model tags now navigate to and open the corresponding A1 model details.
- Scope: Presentation and navigation only. Experiment choices and measured-results status are unchanged.
- Verification: Local responsive, navigation, keyboard and readability checks; human review and hosted GitHub Pages verification remain pending.

## 2026-09-09 - Optimizer support expansion and RMSprop validation

- Tool/model: GitHub Copilot
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 trainer and configuration validation
- Purpose: Expand optimizer support beyond the initially allowed set and reconcile the trainer with the configuration validator, including re-checking RMSprop handling.
- Affected files/sections:
  - `src/dlbench/a1/trainer.py`
  - `src/dlbench/common/config.py`
  - `tests/test_trainer.py`
- Prompt summary: Requested support for `rmsprop` plus several common optimizers, then re-checked the RMSprop behavior to ensure validator and implementation stay aligned.
- AI contribution: Updated trainer optimizer creation to support `rmsprop`, `adagrad`, `adadelta`, `adamax`, and `nadam`, while preserving existing Adam/AdamW/SGD behavior and consistent error messages for unsupported values. Added/updated unit tests to verify `rmsprop` is accepted and unknown optimizers are rejected with the proper error text.
- Student verification: Reviewed the validator and trainer together to ensure the supported set matches the implementation, then ran `python -m unittest tests.test_trainer tests.test_config tests.test_checkpoint` and confirmed all tests passed.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification: PyTorch optimizer APIs for supported optimizers, the project’s validator contract, and the repository test suite.
## 2026-09-08 - Trainer test generation and training-loop coverage

- Tool/model: GitHub Copilot; OpenAI ChatGPT 5.6 Luna
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 trainer implementation
- Purpose: Generate tests for trainer calculations, actual loss reduction, checkpoint evaluation, early stopping, and periodic checkpoint saving.
- Affected files/sections:
  - `src/dlbench/a1/trainer.py`
  - `tests/test_trainer.py`
- Prompt summary: Requested trainer tests that verify calculations using real training loss, evaluate checkpoint behavior, and early stopping/checkpoint frequency logic.
- AI contribution: Added tests for real two-epoch loss reduction using a deterministic model and DataLoader, validation/test checkpoint loader selection, checkpoint configuration and split-hash mismatch rejection, optimizer settings, validation metric calculations, early stopping monitors, patience, and save frequency.
- Student verification: Reviewed the tests against the trainer and checkpoint contracts and ran `python -m unittest -v tests.test_trainer`.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification: Existing trainer, engine, checkpoint, contract, and artifact interfaces; PyTorch optimizer and loss behavior.

## 2026-09-09 - Artifact and provenance design

- Tool/model: OpenAI Codex 5.6 Sol
- Used by: Tạ Tuấn Khải
- Stage: C03 notebook exploration
- Purpose: Design safe run directories and reproducibility artifacts before source implementation
- Affected sections: Run ID validation, JSON writing, source snapshots, Git metadata, split provenance, normalization provenance, and SHA-256 hashing
- Prompt summary: Requested step-by-step explanations and notebook experiments for artifact persistence
- AI contribution: Explained non-overwriting file operations, safe run identifiers, canonical JSON hashing, file hashing, source snapshots, and metadata validation order
- Human verification: Executed each notebook experiment and confirmed the expected success or rejection behavior
- Responsible member: Tạ Tuấn Khải
- Verification sources: Python standard library behavior, temporary-directory experiments, and project configuration files

## 2026-09-09 - Optimizer support expansion and RMSprop validation

- Tool/model: GitHub Copilot
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 trainer and configuration validation
- Purpose: Expand optimizer support beyond the initially allowed set and reconcile the trainer with the configuration validator, including re-checking RMSprop handling.
- Affected files/sections:
  - `src/dlbench/a1/trainer.py`
  - `src/dlbench/common/config.py`
  - `tests/test_trainer.py`
- Prompt summary: Requested support for `rmsprop` plus several common optimizers, then re-checked the RMSprop behavior to ensure validator and implementation stay aligned.
- AI contribution: Updated trainer optimizer creation to support `rmsprop`, `adagrad`, `adadelta`, `adamax`, and `nadam`, while preserving existing Adam/AdamW/SGD behavior and consistent error messages for unsupported values. Added/updated unit tests to verify `rmsprop` is accepted and unknown optimizers are rejected with the proper error text.
- Student verification: Reviewed the validator and trainer together to ensure the supported set matches the implementation, then ran `python -m unittest tests.test_trainer tests.test_config tests.test_checkpoint` and confirmed all tests passed.
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification: PyTorch optimizer APIs for supported optimizers, the project’s validator contract, and the repository test suite.

## 2026-09-10 - Artifact implementation and testing

- Tool/model: OpenAI Codex 5.6 Sol
- Used by: Tạ Tuấn Khải
- Stage: C03 implementation
- Purpose: Implement run-directory creation and reproducibility metadata persistence
- Affected sections: `src/dlbench/common/artifacts.py` and `tests/test_artifacts.py`
- Prompt summary: Requested implementation guidance based on the completed notebook experiments and help diagnosing failing tests
- AI contribution: Reviewed helper functions, identified incomplete metadata writing, suggested validation and serialization order, and explained test and dependency failures
- Human verification: Wrote the implementation, ran the artifact tests and complete unit-test suite, and inspected the Git diff
- Verification results:
  - `python -m unittest discover -s tests -v` — 89 tests run: 86 passed and 3 skipped; no failures or errors
  - The three skips correspond to existing model/split TODOs unrelated to C03
  - `git diff --check` — passed
- Responsible member: Tạ Tuấn Khải
- Verification sources: `tests/test_artifacts.py`, local unit-test output, project configuration files, and Git diff inspection
- Experiment impact: No training runs or benchmark measurements were performed as part of this work

## 2026-09-12 - Trainer tests and scheduler integration

- Tool/model: Gemini 3.1 Pro (High)
- Used by: Nguyễn Anh Khoa
- Stage: Assignment 1 trainer and MLP testing
- Purpose: Add tests for scheduler logic and MLP end-to-end training integration, and fix checkpoint evaluation metadata saving.
- Affected files/sections:
  - `tests/test_trainer.py`
  - `src/dlbench/a1/trainer.py`
- Prompt summary: Requested adding tests to test the scheduler and trainer for the MLP model, and then fix the `save_run_metadata` call in `evaluate_checkpoint` along with adding test coverage for it.
- AI contribution: Added unit tests for various PyTorch learning rate schedulers (`StepLR`, `ExponentialLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `PolynomialLR`) and an end-to-end `smoke=True` integration test for the MLP trainer. Fixed a bug in `evaluate_checkpoint` where `save_run_metadata` lacked a `run_dir` argument by supplying a temporary directory, preventing side effects during evaluation. Added assertions to test that `save_run_metadata` is called correctly.
- Student verification: Verified that the tests passed successfully in the virtual environment without causing file system side effects or metadata corruption. 
- Responsible member: Nguyễn Anh Khoa
- Sources used for verification: Existing trainer logic, Python `tempfile` module documentation, and project `test_trainer.py` file.
