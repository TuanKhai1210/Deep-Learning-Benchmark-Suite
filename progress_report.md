## 1. Data Contracts and Shared Interface

Every model receives a `Batch` - a typed dictionary carrying:
- `images`: a `float32` tensor of shape `[B, 1, 28, 28]`, normalised with training-partition statistics,
- `labels`: an `int64` tensor of shape `[B]`, with class values in `{0, ..., 9}`,
- `sample_ids`: a list of strings uniquely identifying each sample.

Every model must return raw logits of shape `[B, 10]`. Internal flattening, sequence construction, or patch tokenisation is the responsibility within each model without outside preprocessing. Softmax is not before `CrossEntropyLoss`.

The training pipeline produces two structured result types: `EpochMetrics` (loss, accuracy, macro-F1, and sample count for a full epoch) and `EvaluationResult` (wrapping `EpochMetrics` together with a `Predictions` object containing aligned `sample_ids`, `targets`, `predicted_labels`, and per-class `probabilities`).



## 2. Metric Formulation

### 2.1 Classification Metrics

Two metrics are used to assess model quality on a given split: **accuracy** and **macro-averaged F1 score**. Both are implemented in `classification_metrics` in `metrics.py`. Using `zero_division=0`, classes with no predicted instances contribute an F1 of 0. This formulation treats all ten Fashion-MNIST classes equally regardless of their sample frequency in the split.

Macro-F1 is computed over the full set of concatenated predictions from an epoch, not as an average of F1 values over individual batches. The per-batch average introduces a dependency on batch size and class distribution within each batch, which can produce biased estimates especially when the final batch is smaller than the rest. All predictions are therefore accumulated across the entire epoch and `classification_metrics` is called once at the end. This ensures the metric is robust to variations in batch sizing and class distributions.

### 2.2 Parameter Counting

Model capacity is reported through `count_parameters` in `metrics.py`, which returns the total number of scalar parameters (using `parameter.numel()`) and the subset that are trainable (`requires_grad=True`). These counts support comparisons between architectures in the benchmark.

## 3. Checkpoint Selection and Persistence

### 3.1 Selection Criterion

At the end of each epoch, a checkpoint selection decision is made based on validation-set metrics only. This logic is implemented in `is_better` in `checkpoint.py`. The criterion is a strict lexicographic order over three quantities:

1. **Macro-F1 (maximise)**: the primary criterion, reflecting per-class discrimination quality.
2. **Validation loss (minimise)**: the first tiebreaker when F1 scores are equal, capturing calibration differences.
3. **Epoch (minimise)**: the final tiebreaker, preferring earlier-converging models when all other metrics are identical.

Candidates with non-finite metric values are unconditionally rejected to guard against pathological training states being promoted to the best checkpoint. When no incumbent exists (first epoch), any valid candidate is accepted.

### 3.2 Checkpoint Payload Schema

Two checkpoint files are maintained per training run. **`best.pt`** is written by `save_checkpoint` whenever `is_better` returns `True`, and contains the model state dict, epoch number, complete resolved configuration, validation metrics, run seed, schema version, and hashes of the data split and normalisation statistics. **`last.pt`** is the full resumable state written at the configured save frequency and at training termination. In addition to everything in `best.pt`, it includes the optimizer state dict, scheduler state dict, the complete RNG state across all frameworks, and the early-stopping counters.

The `split_hash` and `statistics_hash` fields included in every checkpoint serve as provenance anchors: they allow any future evaluation or resumption to verify that the data pipeline is identical to the one used during training.

### 3.3 Atomic Write Protocol

Checkpoints are written using an atomic rename strategy in `save_checkpoint`: the payload is first written to a temporary file, then moved into place via `os.replace()`. This guarantees that an interrupted write cannot corrupt an existing valid checkpoint. 

Loading via `load_checkpoint` performs symmetric schema validation, ensuring that any checkpoint opened for resumption or evaluation is structurally complete before any weights are consumed.


## 4. Training and Evaluation Engine

### 4.1 Single-Epoch Training

The training pass for a single epoch is implemented in `train_one_epoch` in `engine.py`. It follows the standard mini-batch stochastic gradient descent procedure, with the model set to training mode to enable stochastic regularisation components such as Dropout. For each batch:

1. Images and labels are moved to the target device.
2. Accumulated gradients are cleared (`optimizer.zero_grad()`).
3. A forward pass produces raw logits.
4. Cross-entropy loss is computed from the logits and integer labels.
5. Gradients are computed via backpropagation (`loss.backward()`).
6. The optimizer updates the model parameters (`optimizer.step()`).

Batch loss values are accumulated as a sample-weighted sum and divided by the total number of samples at the end of the epoch. This is necessary to correctly handle the final batch, which may be smaller than the configured batch size. A naive mean of per-batch losses would weight the final batch equally to full batches, introducing a systematic bias. The function returns an `EpochMetrics` object computed by calling `classification_metrics` over all accumulated predictions.

### 4.2 Evaluation Pass

The evaluation pass is implemented in `evaluate_epoch` in `engine.py`, operating under `model.eval()` and `torch.inference_mode()` so that no parameters are modified. Beyond replicating the loss and metric computation of the training pass, it additionally collects `sample_ids` - preserved in the exact order they were yielded by the loader - and per-class probabilities obtained by applying softmax to the raw logits. The probabilities are stored in the `Predictions` object for downstream analysis and are never fed back to the loss function. Strict alignment between `sample_ids` and predictions is enforced at the batch level to prevent silent mismatches between a sample and its model output.

The function returns an `EvaluationResult` containing both the summary `EpochMetrics` and the full `Predictions` object, enabling both aggregate reporting and sample-level analysis.

### 4.3 Train/Eval Mode Correctness

The distinction between training and evaluation mode is enforced at the trainer level. `train_one_epoch` unconditionally sets the model to training mode on entry; `evaluate_epoch` unconditionally sets it to evaluation mode. This ensures that regularisation layers behave correctly regardless of the model's state at the point of the call and the model's parameters are not modified during evaluation.


## 5. Training Procedure

### 5.1 Initialisation and Reproducibility

Before any model parameters are initialised or data is loaded, the global random state is seeded deterministically by `seed_everything` from `common/reproducibility.py`, covering Python's `random`, NumPy, and all PyTorch RNG states. Deterministic cuDNN operations are enabled where available.

A unique run identifier is generated by `generate_run_id` and a dedicated output folder is created before training begins. The resolved configuration and environment metadata are written to disk at this point so that every run's exact settings remain recoverable even if the source config files are later modified. Each run folder contains the resolved config, run metadata, the best and last checkpoints, per-epoch training history, and final validation predictions and metrics.

### 5.2 Smoke vs. Main Runs

Two execution modes are supported. In smoke mode, the epoch budget and batch size are capped to enable rapid end-to-end correctness checks without a full training budget. Main runs undergo strict configuration validation via `validate_config` with no imposed limits beyond those in the config.

### 5.3 Epoch Loop

Each epoch calls `train_one_epoch` followed by `evaluate_epoch` on the validation loader. After evaluation, the following actions are taken in order:

1. A history row (train/val loss, accuracy, macro-F1, learning rate, and wall-clock time) is appended to `history.csv` via `append_history`.
2. The learning rate scheduler, if configured, is stepped. `ReduceLROnPlateau` receives the monitored metric (macro-F1 if `mode=max`, otherwise loss is used); all others step unconditionally.
3. `is_better` is applied to determine whether to overwrite `best.pt` via `save_checkpoint`.
4. The early stopping counter is updated; if the monitored metric has not improved by at least `min_delta` for `patience` consecutive epochs, training terminates.
5. `last.pt` is written via `save_checkpoint` at the configured save frequency, on the final epoch, or on early-stop termination.

### 5.4 Training Resumption

When a `resume_from` path is provided, `fit` loads the checkpoint via `load_checkpoint` and verifies that both the data provenance hashes and the key configuration sections (`protocol`, `model`, `preprocessing`, `training`) match the current run. This ensures that resumption is only permitted when the data pipeline and model architecture are identical to the original run. The full RNG state is then restored, the history file is correctly truncated to the checkpoint epoch, and training continues from `epoch + 1` using the historical best-model threshold as the benchmark for subsequent checkpoint selection.

### 5.5 Optimizer and Device Support

The optimizer is constructed by `get_optimizer` from the `training.optimizer` field of the configuration. Supported optimizers are: **Adam**, **AdamW**, **SGD** (with optional momentum), **RMSprop**, **Adagrad**, **Adadelta**, **Adamax**, and **NAdam**. Device selection is handled by `resolve_device`, which reads `run.device` from the configuration or auto-detects hardware following the priority order CUDA -> MPS -> CPU.


## 6. Checkpoint-Based Evaluation Protocol

After training, the selected best checkpoint is evaluated on either the validation or test split using `evaluate_checkpoint` in `trainer.py`. Before loading any weights, the function verifies that the supplied configuration matches the checkpoint's stored configuration and that the independently computed data provenance hashes match those stored in the payload. This ensures that the evaluation is performed on the same data and with the same preprocessing as the original training run.

The test split is inaccessible during the development cycle and can only be requested explicitly outside of smoke mode. This prevents test-set leakage into model selection decisions. The best checkpoint is chosen solely on performance on the validation set, and the test set is evaluated at most once per model after all architectural and hyperparameter decisions are fixed.

The function reconstructs the model loads the saved weights, and calls `evaluate_epoch` on the chosen loader. The returned `EvaluationResult` is structurally identical to the one produced during training, enabling direct comparison of validation metrics computed epoch-by-epoch with those recomputed from the final checkpoint.

## 7. MLP Architecture

### 7.1 Model Design

`MLPClassifier` in `models/mlp.py` is a fully-connected feedforward network for 10-class image classification. The architecture is constructed dynamically from a configuration dictionary with no hard-coded layer sizes, following this topology for a depth-`n` network:

```
Flatten -> [Linear(d_{i-1}, d_i) -> Activation -> Dropout(p)?] x n -> Linear(d_n, 10)
```

An `nn.Flatten` layer at the front of the network converts `[B, 1, 28, 28]` image tensors to `[B, 784]` flat representations internally, keeping the model's external interface consistent with the shared `Batch` contract. The activation function is selected from a supported registry (`relu`, `gelu`, `tanh`, `sigmoid`, `leakyrelu`) via a string parameter. Dropout is inserted after each hidden activation only when the rate is strictly positive. The output head produces 10 raw logits with no activation applied.

### 7.2 Contribution of Hidden Nonlinearities

For a flattened image $x\in\mathbb{R}^{784}$, the Linear classifier computes $z = Wx+b,$ where $z\in\mathbb{R}^{10}$ contains the class logits. Consequently, the boundary between any two classes $i$ and $j$ is the hyperplane $(w_i-w_j)^{\mathsf T}x+(b_i-b_j)=0$. This restricts each pairwise class separation to a single linear boundary in pixel space. Increasing the number of consecutive linear layers without adding non-linearity would not resolve this, as the composition can always be rewritten as one linear transformation.

The MLP instead computes a hierarchy of representations through a point-wise nonlinear activation $\sigma$. For instance:  $$ \begin{aligned}
    h_1 &= \sigma(W_1x+b_1),\\
    h_2 &= \sigma(W_2h_1+b_2),\\
    z   &= W_3h_2+b_3. \end{aligned} $$

The nonlinear activation $\sigma$ introduces input-dependent transformations at each layer: different inputs produce different intermediate representations, breaking the global linearity of the network. As a result, the complete network represents a nonlinear function of the input rather than a single global hyperplane, and pairwise class boundaries in the original input space can be curved or piecewise-defined depending on the chosen activation.

This additional capacity is useful for Fashion-MNIST because class evidence often depends on combinations of pixels rather than on independent pixel intensities. A first-layer unit can respond to a weighted combination associated with a boundary, opening, sleeve, sole, or other distributed intensity pattern. The following layer can combine several of these responses conditionally; for example, evidence at the upper and side regions of an image may jointly support a shirt-like class, while a different combination may support a bag or ankle boot. Such feature interactions and disjoint decision regions cannot, in general, be represented by a single Linear classifier.

The nonlinear MLP therefore provides greater representational capacity while retaining a simple fully connected architecture. However, flattening maps nearby and distant pixels into an undifferentiated vector. The model does not explicitly know that adjacent entries correspond to neighboring pixels, and it does not share a feature detector across different positions. It must learn these spatial relationships separately from data, which increases the number of parameters significantly. Its extra capacity can also increase overfitting, so dropout and validation-based checkpoint selection are used to regularise and select the implemented model.