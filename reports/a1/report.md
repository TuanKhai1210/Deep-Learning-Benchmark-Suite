# Assignment 1 Report

Status: **outline only — no measured results**.

Institution/course/instructor: TODO. Group/member names and IDs: TODO. Submission revision/date: TODO.

## 1. Problem and Data Description

### 1.1 Objective

TODO: state the image-classification question and what a fair model-family comparison should reveal beyond accuracy.

### 1.2 Dataset and provenance

TODO: Fashion-MNIST source/license, labels, shape, official counts, access method and dependency versions. Cite original sources actually consulted.

### 1.3 Exploratory data analysis

TODO: measured class distribution, representative images, imbalance/quality observations and relevant dataset limitations.

### 1.4 Split and leakage controls

TODO: actual train/validation/test counts, stratification, split seed, saved indices/identity, checks for overlap and train-only preprocessing statistics. Clearly distinguish proposals from implemented decisions.

## 2. Methodology

### 2.1 Shared pipeline and preprocessing

TODO: data flow, tensor/label contracts, normalization statistics, train-only augmentation and deterministic evaluation transforms.

### 2.2 Models and representations

TODO: explain each implemented architecture, parameterization and classifier output.

- Linear: flattened input and decision function.
- MLP: hidden layers, activation and regularization.
- CNN: self-designed convolution/pooling and spatial inductive bias.
- LSTM or GRU: selected cell, timestep representation, hidden state and aggregation.
- Transformer: token/patch construction, projection, positional encoding, attention and aggregation.

### 2.3 Training, tuning and selection

TODO: optimizers, LR/schedulers, batch size, precision, epoch cap, early stopping, tuning budgets, run seeds and validation-based checkpoint rule. Justify differences across models.

### 2.4 Evaluation and timing

TODO: accuracy, whole-split macro-F1, loss aggregation, parameter counts, training-time boundary, inference scope, warm-up/repetitions/synchronization and hardware.

### 2.5 Reproducibility

TODO: protocol ID, code revision, exact environment, commands, artifact locations and checkpoint/reconstruction instructions.

## 3. Implementation Results

### 3.1 Main comparison

Experiments are in preparation. Fill this table only from verified experiment artifacts; state units and seed aggregation.

| Model | Test accuracy | Test macro-F1 | Parameters | Training time | Inference time | Run/config reference |
|---|---|---|---|---|---|---|
| Linear | TODO | TODO | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | TODO | TODO | TODO |
| CNN | TODO | TODO | TODO | TODO | TODO | TODO |
| LSTM or GRU | TODO | TODO | TODO | TODO | TODO | TODO |
| Transformer | TODO | TODO | TODO | TODO | TODO | TODO |

### 3.2 Learning dynamics

TODO: train/validation curves, actual epochs, convergence, overfitting/underfitting and checkpoint selection evidence.

### 3.3 Error analysis

TODO: confusion matrices in a consistent class order; correct/incorrect examples with sample IDs; class-level confusions and cautious explanations supported by evidence.

### 3.4 Representation and inductive bias

TODO: compare flattened, spatial, sequential and token representations; discuss how locality, weight sharing, recurrence and attention relate to measured behavior. Do not confuse architectural inductive bias with class imbalance alone.

### 3.5 Trade-offs and limitations

TODO: performance/compute/size trade-offs, hardware and tuning constraints, variance, dataset limits and threats to fair comparison. Keep optional ablations separate from the main protocol.

### 3.6 Conclusion

TODO: answer the original comparison question using actual findings, without claiming general superiority from one small dataset.

### 3.7 Contributions and AI disclosure

TODO: list each member's implemented, reviewed, written and presented work. Initial repository/interface assistance: OpenAI Codex, 2026-09-04, exact model identifier not recorded; human verification pending. Extend with actual AI usage, affected sections and verification performed; keep consistent with `AI_USAGE.md`.

### 3.8 References and artifact index

TODO: verified primary sources; code revision, configs, logs, checkpoint/reconstruction instructions, slides and video links. Do not insert links to artifacts that do not yet exist.
