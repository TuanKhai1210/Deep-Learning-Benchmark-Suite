# A1 experiment contract

Status: **draft for team review**. This document explains the shared protocol in `configs/a1/protocol.py`. Team responsibilities and the four training seeds are agreed; the remaining proposed defaults are not instructor-mandated values and have not yet been approved or measured.

## Approval record

| Field | Value |
|---|---|
| Protocol ID | `a1-v0` (draft) |
| Owners / reviewers | A: Nguyễn Hạo Thiên — data + Linear/CNN, reviewed by B; B: Nguyễn Anh Khoa — engine + MLP/RNN, reviewed by C; C: Tạ Tuấn Khải — common tools + Transformer/integration, reviewed by A |
| Agreed training seeds | `[36, 69420, 67, 69]`; default development run seed `36` |
| Approved by / date | TODO |
| Measured environment | TODO |
| Change history | 2026-09-04: record agreed roles/seeds and Python configuration; no approved experiment runs |

## Scope

Fashion-MNIST is the main A1 dataset. Required model families are Linear, MLP, custom CNN, LSTM or GRU, and Transformer. MNIST is only an optional debug aid; CIFAR-10 and other extensions must not displace required work.

Prioritize EDA, Dataset/DataLoader, train/validation, Linear and MLP for the Draft. Complete all five families and the full comparison for the Final.

## Data and preprocessing

| Decision | Proposed policy |
|---|---|
| Source | Fashion-MNIST through torchvision; record provenance and dependency versions |
| Split | Official training set → stratified 50,000 train / 10,000 validation; official 10,000 test unchanged |
| Split seed | Proposed `42`, independent from the agreed model run seeds |
| Persistence | Generate indices once, commit them under `configs/a1/splits/`, record identity/checksum |
| Input | Grayscale 28 × 28; normalized float32 `[B, 1, 28, 28]` |
| Targets | int64 `[B]`, class indices 0–9 with one recorded class order |
| Normalization | Mean/std measured only on the unaugmented training subset; currently unknown |
| Augmentation | Draft proposal: random crop size 28, padding 2 on train only; review before freezing |
| Validation/test | Deterministic preprocessing; no random augmentation |
| Sample identity | Record source and original sample ID to avoid train/test ID ambiguity |

No test-based tuning, train/validation overlap, silent sample removal, or statistics fitted to validation/test. Debug runs may omit augmentation but must not be mislabeled as runs from the final frozen protocol.

## Interfaces and representations

Every model accepts the same image tensor and returns raw logits `[B, 10]`; internal adapters handle flattening, rows, or patches. CrossEntropyLoss receives logits, not softmax probabilities.

- Linear/MLP: flattened image.
- CNN: spatial image input with self-designed architecture.
- RNN: choose LSTM or GRU and record the row/column/patch sequence, timestep input size, hidden size, and aggregation rule.
- Transformer: record patch/token construction, projection, positional encoding, attention dimensions, and classification aggregation.

Architecture parameters in model configs are starting proposals. Tune with validation within the approved budget. Configurations are `.py` files containing a literal `CONFIG` dictionary, without imports or runtime side effects. Each model config references the shared protocol with `"protocol_file": "../protocol.py"` in its dictionary. Do not copy the shared split or metric policy into five independent files.

## Reproducibility and training policy

- Record Python/PyTorch/torchvision/CUDA versions, hardware, precision, determinism settings, code revision, split ID, and effective config.
- Seed Python, NumPy, PyTorch and DataLoader workers/generators as applicable. Seeds do not promise bitwise identity across all machines/versions.
- Agreed run seeds: `[36, 69420, 67, 69]`. Default development seed: `36`. The main comparison plans five model families × four seeds = 20 runs, excluding tuning/debugging. Reinitialize each model for every seed and keep the same saved split. If resources require changing this commitment, obtain team agreement and version the change before publishing results.
- Train loader shuffles; validation/test loaders do not shuffle or drop samples.
- Fix batch size, epoch cap, early-stopping semantics, tuning trials/search space and scheduler policy after smoke tests, before full runs.
- Different optimizers/LRs may be appropriate. Fairness requires a transparent tuning budget and common evaluation, not blindly identical hyperparameters.
- Select configurations using validation only; never select the best test seed.

## Metrics, checkpoint and timing

Accuracy is total correct divided by total samples. Macro-F1 is computed from predictions for the full split with all 10 labels and `zero_division=0`, not averaged across batch F1 values. Loss aggregation is weighted by sample count. Logs use fractions for accuracy/F1; reports must label any percentage conversion.

Checkpoint selection proposal: highest validation macro-F1; ties resolved by lower validation loss, then earlier epoch. `best.pt` is for final evaluation; `last.pt` is separate if resume is supported. Record the config, epoch, split, seeds, preprocessing, code revision and validation score with the checkpoint. Save optimizer/scheduler/RNG state when exact training continuation is supported.

Report total/trainable parameters, training time, inference time, actual epochs, train/validation curves, confusion matrix, correct/incorrect examples, and qualitative analysis. Report accuracy and macro-F1 as mean and standard deviation across all four agreed seeds without cherry-picking; retain the per-seed results and state the standard-deviation convention.

Before timing, approve the hardware, batch size, precision, warm-up/repetition counts and aggregation. The proposed inference scope is forward-only; exclude or separately measure loading and transfers. Use evaluation/no-gradient inference and appropriate CUDA synchronization. Amortized time per image in a batch is not batch-one latency. Define whether training timing includes validation/checkpoint/logging; exclude download and EDA, and report tuning cost separately.

## Artifacts and evidence

Each run has a unique ID and stores effective config, metadata, history, metrics, predictions/sample IDs and checkpoint. Source configuration is Python; the resolved settings may be serialized as JSON for run evidence. Keep large data/runs/checkpoints out of Git; store small reviewed publication tables/figures under `docs/assets/a1/`. Every report result must be traceable to an actual run.

## Required before a frozen main comparison

- [x] Owners/reviewers and four run seeds recorded.
- [ ] Environment and hardware selected.
- [ ] Split created and independently checked for counts, class balance and overlap.
- [ ] Mean/std measured and recorded from the correct subset.
- [ ] Augmentation reviewed and fixed.
- [ ] Shared interfaces, evaluator and checkpoint behavior tested.
- [ ] Budgets for the four-seed plan, early stopping and per-model tuning policy approved.
- [ ] Timing procedure and hardware confirmed.
- [ ] Artifact tracing and checkpoint reload checked.
- [ ] Config strict validation succeeds after genuine decisions, not dummy values.
- [ ] Protocol ID, `frozen` status and approvers recorded.

Unresolved values do not block writing interfaces, preparing data or running development smoke tests. They do block labeling experiments as the final approved comparison.

After freezing, review common-protocol changes, increment the ID and rerun affected comparisons. Do not mix incompatible protocols in one unlabeled table.
