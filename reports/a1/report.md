# Assignment 1 Report

Status: **methodology draft — experiment settings and measured results pending**.

Submission rule (handbook revision 14 September 2026): export the Final as `<GroupName>_A1_Report.pdf`, with Group Name matching `GroupRegistration` exactly. The designated representative submits this PDF on LMS; a Pages link is not a substitute. Group Name and submitter remain to be confirmed. Draft may be linked from the landing page; check milestone-specific LMS announcements. See [v2 checklist](../../docs/handbook-v2-update.md).

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

This methodology focuses on the group's MLP and recurrent classifiers and on the shared training, evaluation, and checkpoint logic used to run them. It describes both the model definitions and the reasoning behind the implementation choices. Where the recurrent architecture is configurable, the method is written to cover the supported alternatives rather than assume that the current example configuration is the final experimental model. Training hyperparameters and measured outcomes will be supplied with the completed experiments.

### 2.1 Shared input and output contract

The models are integrated through a common batch interface. A batch contains:

| Field | Representation | Meaning |
|---|---|---|
| `images` | `float32 [B, 1, 28, 28]` | Normalized grayscale images |
| `labels` | `int64 [B]` | Integer class indices from 0 through 9 |
| `sample_ids` | `list[str]` | Stable identifiers used to align predictions with examples |

Here, \(B\) denotes the batch size. Both classifiers return a tensor of raw logits with shape `[B, 10]`, one score for each of the ten Fashion-MNIST classes. The models perform their own internal representation conversion: the MLP flattens each image, while the RNN reshapes or permutes the image into an ordered sequence. This shared external interface lets the trainer use the same loss, optimizer orchestration, validation pass, metrics, and checkpoint workflow for either architecture.

Logits are unnormalized class scores. A softmax transformation can convert them to probabilities for interpretation, but cross-entropy loss is defined to consume logits directly. Keeping softmax out of the loss path avoids redundant transformations and follows the expected interface of the PyTorch cross-entropy implementation.

### 2.2 Multilayer perceptron

#### Architecture and parameterization

The MLP is a fully connected feed-forward classifier constructed from a model-configuration mapping. Its general structure is

```text
Flatten -> [Linear -> Activation -> Dropout (optional)] x n -> Linear -> 10 logits
```

The input image is flattened internally from `[B, 1, 28, 28]` to `[B, 784]`. The configuration supplies the input dimension, number of classes, a non-empty list of hidden-layer widths, activation name, and dropout probability. For hidden widths \(d_1,\ldots,d_n\), with \(d_0=784\) and \(d_{n+1}=10\), each affine layer computes
\[
a_i = W_i h_{i-1} + b_i,\qquad h_i = \sigma(a_i),\quad i=1,\ldots,n,
\]
and the output layer computes
\[
z = W_{n+1}h_n+b_{n+1}.
\]
The output \(z \in \mathbb{R}^{10}\) contains class logits. There is deliberately no output softmax in the model.

The implemented activation registry supports ReLU, GELU, tanh, sigmoid, and Leaky ReLU. The chosen activation is applied independently to each hidden unit. Nonlinear activation is essential: without it, a composition of fully connected linear layers is itself equivalent to one linear map, so depth alone would not give the classifier nonlinear decision boundaries. With nonlinearities, later layers can form conditional combinations of earlier features and represent more complex partitions of pixel space.

The current MLP configuration is an explicit starting example with hidden widths `[256, 128]`, ReLU, and dropout probability \(p=0.1\). Its resulting topology is
\[
784 \rightarrow 256 \rightarrow 128 \rightarrow 10.
\]
For this configuration, the hidden activations before dropout are
\[
u_1=\operatorname{ReLU}(W_1x+b_1),\qquad
u_2=\operatorname{ReLU}(W_2u_1+b_2),
\]
and training applies dropout to each hidden activation before passing it to the next layer. The classifier contains 235,146 trainable scalar parameters for this configuration: \(784 \times 256 + 256\) in the first layer, \(256 \times 128 + 128\) in the second, and \(128 \times 10 + 10\) in the output layer. This count is configuration-specific, not a fixed property of all MLPs supported by the implementation.

#### Activation and regularization interpretation

ReLU is defined as \(\operatorname{ReLU}(u)=\max(0,u)\). It preserves positive responses and sets negative responses to zero, producing a piecewise-linear nonlinear network. GELU smoothly weights inputs according to their magnitude; tanh and sigmoid bound activations to finite ranges; Leaky ReLU retains a small slope for negative inputs. Because these choices can affect optimization and representation, the activation name is an explicit model parameter rather than a hidden implementation assumption.

Dropout randomly masks hidden activations during training, with each activation independently retained with probability \(1-p\) under the standard dropout convention. This noise discourages units from relying too strongly on specific co-activations and can reduce overfitting. Dropout is active in training mode and disabled in evaluation mode, so validation and test predictions are deterministic for fixed weights and inputs. A zero dropout rate omits dropout layers entirely. The model validates that at least one hidden layer is supplied, that hidden widths are positive, that the dropout value is in the accepted range, and that the requested activation exists in the registry.

#### Representation and inductive bias

Flattening gives each dense layer access to all 784 pixel positions, which allows global interactions to be learned directly. However, the model does not explicitly encode which pixels are neighbors, nor does it share a learned detector across spatial locations. A pattern appearing in different positions may therefore require distinct weights. This makes the MLP a useful nonlinear dense baseline, while distinguishing its representation from spatially structured models.

### 2.3 Configurable recurrent image classifier

#### Sequence construction

The RNN converts each image into an ordered series of row or column vectors. For a grayscale image \(X \in \mathbb{R}^{H \times W}\):

| Representation | Sequence length \(T\) | Input size \(D\) per timestep | Construction |
|---|---:|---:|---|
| Rows | \(H\) | \(W\) | \(x_t = X_{t,:}\) |
| Columns | \(W\) | \(H\) | \(x_t = X_{:,t}\) |

For Fashion-MNIST, \(H=W=28\), so either orientation produces a length-28 sequence with 28 values at each timestep. The difference is the spatial direction through which evidence is presented to the recurrent network. For a batch, removing the one-channel dimension produces `[B, 28, 28]`; column representation transposes the two spatial axes before the recurrent layer. The input size must agree with the selected orientation and image dimensions.

Each timestep is therefore a complete image row or column, not one individual pixel and not the whole flattened image. The ordered sequence allows the recurrent state to carry information from earlier rows or columns to later ones. This gives the model an explicit scan direction and an inductive bias for accumulating evidence across that direction.

#### Recurrent cell: GRU or LSTM

The recurrent cell is configurable as a GRU or an LSTM. Both process one feature vector at a time and update a hidden representation based on the current input and state from preceding timesteps. Their gating mechanisms provide paths for retaining or discarding information over the sequence.

For a GRU, the reset gate \(r_t\) controls how the preceding hidden state contributes to a candidate state, while the update gate \(z_t\) controls how much of the prior state is carried into the new state. A common GRU formulation is
\[
\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1}+b_r),\\
z_t &= \sigma(W_z x_t + U_z h_{t-1}+b_z),\\
\widetilde{h}_t &= \tanh(W_h x_t+U_h(r_t\odot h_{t-1})+b_h),\\
h_t &= z_t\odot h_{t-1}+(1-z_t)\odot\widetilde{h}_t,
\end{aligned}
\]
where \(\sigma\) is the sigmoid function and \(\odot\) denotes element-wise multiplication. Gate naming and update-equation conventions can vary between references and libraries; the essential function is learned control over state retention and replacement.

An LSTM maintains two state vectors at each timestep: a hidden state \(h_t\), exposed to the next layer or classifier, and a cell state \(c_t\), which carries longer-term memory. Its input gate controls candidate information written to memory, its forget gate controls information retained from \(c_{t-1}\), and its output gate controls how much of the cell state is exposed as \(h_t\). Conceptually,
\[
\begin{aligned}
i_t &= \sigma(W_i x_t+U_i h_{t-1}+b_i),\\
f_t &= \sigma(W_f x_t+U_f h_{t-1}+b_f),\\
o_t &= \sigma(W_o x_t+U_o h_{t-1}+b_o),\\
\widetilde{c}_t &= \tanh(W_c x_t+U_c h_{t-1}+b_c),\\
c_t &= f_t\odot c_{t-1}+i_t\odot\widetilde{c}_t,\\
h_t &= o_t\odot\tanh(c_t).
\end{aligned}
\]
The hidden and cell states are distinct: the classifier uses the hidden state, not the LSTM cell memory itself.

#### Depth and directionality

The RNN configuration exposes the cell type, input size, hidden size, number of recurrent layers, row/column representation, bidirectionality, and recurrent dropout. For an \(L\)-layer unidirectional recurrent network, each layer processes the sequence in order and passes its timestep outputs to the next layer. The final hidden representation for classification is taken from the last recurrent layer.

A bidirectional network contains a forward recurrence and a reverse recurrence. At each layer, the forward component reads \(x_1,\ldots,x_T\); the reverse component reads \(x_T,\ldots,x_1\). Their hidden representations are concatenated, allowing the representation of the sequence to incorporate context from both scan directions. If the per-direction hidden size is \(H_d\), the final classifier input dimension is \(H_d\) for a unidirectional model and \(2H_d\) for a bidirectional model. The output layer's dimensions are set accordingly; the number of layers changes the recurrent stack but does not change this final feature width.

The implementation handles the distinct PyTorch return structures correctly. A GRU returns the sequence output and final hidden state. An LSTM returns the sequence output and a pair containing final hidden and cell states. The classifier takes the hidden state from the top recurrent layer; for a bidirectional model, it concatenates the top layer's forward and reverse states. It does not substitute the cell state for the LSTM hidden state or average all timestep outputs. Recurrent dropout is applied between recurrent layers by the underlying PyTorch recurrent module; with a single recurrent layer, there is no inter-layer connection on which to apply that dropout.

The current configuration is only one valid instance: a one-layer, unidirectional GRU with row representation, input size 28, and hidden size 128. It should not be read as fixing the final experimental choice. The final report will identify the actual cell, sequence orientation, layer count, hidden size, directionality, and dropout setting used for each reported run.

### 2.4 Single-epoch training mechanics

Training is divided into a single-epoch engine and a multi-epoch trainer. This separation keeps model definitions independent of optimization orchestration and lets both MLP and RNN use the same training function. For every training epoch:

1. The model is placed in training mode using `model.train()`. This enables training-specific behavior such as MLP dropout.
2. The loader yields a batch containing images and labels. These tensors are moved to the selected compute device.
3. The optimizer's accumulated gradients are cleared.
4. The model performs a forward pass and returns `[B, 10]` logits.
5. Cross-entropy loss compares these logits with the integer target labels.
6. Backpropagation computes parameter gradients, and the optimizer performs an update.
7. The pre-update batch logits are converted to predicted class indices with `argmax` for the epoch's classification metrics.

No softmax is applied before cross-entropy. Training loss is accumulated as batch mean loss multiplied by batch size, then divided by the total number of processed examples. If the last batch is smaller than the others, this sample weighting prevents it from receiving the same aggregate weight as a full batch. Targets and predictions are accumulated across the epoch; accuracy and macro-F1 are calculated once over this complete collection, rather than by averaging batch metrics. The epoch result records mean loss, accuracy, macro-F1, and sample count.

### 2.5 Multi-epoch run orchestration and configuration

Before constructing a new run, the trainer validates the configuration and seeds Python, NumPy, and PyTorch random-number generators using the configured run seed. It then builds the data loaders and model, resolves the requested device, transfers the model, and constructs the optimizer, optional scheduler, and cross-entropy criterion. Automatic device selection follows CUDA, then Apple MPS, then CPU; a device can also be supplied explicitly in the run configuration.

Optimizer choice is configuration-controlled. The implementation supports Adam, AdamW, SGD (with momentum), RMSprop, Adagrad, Adadelta, Adamax, and NAdam. Shared optimizer settings include learning rate and weight decay, while optimizer-specific values are read when relevant. Optional learning-rate schedulers include step decay, exponential decay, cosine annealing, reduce-on-plateau, and polynomial decay. Their behavior and numerical settings are configuration values; none of these options implies a single fixed optimizer or schedule for every architecture.

At each epoch the trainer calls the shared training pass, then evaluates on validation. It appends training and validation loss, accuracy, macro-F1, learning rate, and epoch duration to the run history. Epoch duration is measured around the training and validation passes. The configured epoch limit is an upper bound; early stopping can end a run sooner. The report of each experiment must therefore use the actual epoch count and settings saved with that run, not only the planned maximum.

Smoke runs and main runs have different purposes. A smoke run is a correctness check, is marked separately from benchmark runs, and caps the epoch budget at ten and batch size at sixteen (using those values if the supplied values are not positive). It is not evidence for Fashion-MNIST benchmark results. Main runs use strict configuration validation and their approved full settings. Final training values—optimizer, learning rate, batch size, scheduler, epoch cap, patience, `min_delta`, device, and seed—will be stated with the relevant run results once frozen.

### 2.6 Validation, metrics, and prediction records

After every training epoch, `evaluate_epoch` switches the model to evaluation mode and executes the validation pass inside `torch.inference_mode()`. This disables gradient tracking and parameter updates and ensures training-only dropout is inactive. For each validation batch, the evaluator calculates logits and cross-entropy loss, determines predicted labels, and applies softmax only to obtain probabilities for later analysis. It also retains the sample IDs in loader order. The evaluator checks that the IDs are present and that the number of IDs matches the batch size so that predictions cannot silently become detached from their examples.

The returned evaluation record contains summary epoch metrics and a prediction record with aligned sample IDs, target labels, predicted labels, and one probability per class. This makes the evaluation useful both for aggregate scores and for downstream error inspection, such as identifying correct or incorrect examples or constructing a confusion matrix.

The metric definitions are:

- **Accuracy:** number of correctly classified examples divided by the total number of examples.
- **Per-class F1:** for class \(c\), \(F1_c=2P_cR_c/(P_c+R_c)\), where \(P_c\) is precision and \(R_c\) is recall for that class.
- **Macro-F1:** arithmetic mean of the class-specific F1 values for all ten labels:
  \[
  \operatorname{MacroF1}=\frac{1}{10}\sum_{c=0}^{9}F1_c.
  \]

The metric implementation fixes the label set to classes 0–9 and uses `zero_division=0`, so a class with no predicted examples has an F1 contribution of zero rather than an undefined value. Macro-F1 is computed from predictions concatenated over the entire split. Unlike accuracy, which weights each example equally, macro-F1 gives each class equal weight in the final average and therefore exposes performance differences among classes. Loss is averaged using sample counts, not by taking an unweighted mean of batch means. The same definitions are used for epoch training metrics, validation metrics, and checkpoint evaluation.

The parameter-count utility reports both total scalar parameters and trainable scalar parameters. It sums the number of elements in each model parameter tensor, counting only parameters with `requires_grad=True` for the trainable total. These are model-size/capacity descriptors and are not estimates of runtime or compute cost.

### 2.7 Checkpoint selection, persistence, and test-set separation

Checkpoint selection is based only on validation results. Candidates are compared lexicographically:

1. Higher validation macro-F1 is preferred.
2. If macro-F1 is equal, lower validation loss is preferred.
3. If both are equal, the earlier epoch is preferred.

Candidates with non-finite validation macro-F1 or loss are rejected. This selection order makes the chosen checkpoint reflect the primary classification metric while using loss and epoch only as deterministic tie-breakers.

Early stopping is distinct from best-checkpoint selection. The configured early-stopping monitor can be validation macro-F1 or validation loss. An improvement must exceed the configured `min_delta`; otherwise the consecutive no-improvement counter increases. If positive patience is reached, training stops. Thus, the checkpoint ordering (macro-F1, loss, earlier epoch) and the early-stopping monitor/patience serve different purposes and must not be conflated.

Two checkpoint roles are maintained. `best.pt` stores the selected model state and the run context needed to interpret it, including epoch, resolved configuration, validation metrics, run seed, schema version, data split and normalization provenance hashes, and code revision. `last.pt` is written at the configured save frequency and at the end of training or early stopping; it includes the optimizer and scheduler states, random-number-generator state, and early-stopping counters so that training can continue. Checkpoint writes use a temporary file followed by an atomic replace, reducing the risk that an interrupted write leaves a partially written checkpoint in place.

When resuming, the trainer checks that the saved split and normalization hashes and relevant configuration sections match the current run. It restores model, optimizer, scheduler, Python/NumPy/PyTorch RNG states, and the training DataLoader generator state, then continues from the next epoch. Existing history is checked and truncated to the checkpoint epoch before additional rows are written, preserving a consistent epoch sequence.

Checkpoint evaluation reconstructs the model from the supplied configuration and checks compatibility with the checkpoint, including protocol, data, preprocessing, evaluation, checkpoint policy, and model settings. It independently verifies split and normalization provenance before loading the requested validation or test loader. Validation is the default evaluation split. Official-test evaluation requires an explicit `--allow-test` acknowledgement and is prohibited in smoke mode; the test set is reserved until model and configuration choices are fixed. This prevents test metrics from entering epoch selection, early stopping, or tuning.

The above sections specify the implemented method and configurable behavior. Final experiment parameters, observed training curves, selected epochs, metric values, parameter totals for the selected RNN configuration, and timing results will be inserted only after the associated runs are complete and verified.

## 3. Implementation Results

Complete this section from verified experiment artifacts. Replace every `TODO` and image placeholder, distinguish benchmark runs from smoke runs, and report deviations or exclusions. The held-out test set must not be used for model selection.

### 3.1 Completed Runs and Evidence Coverage

Summarize which planned models and seeds completed, identify any excluded runs, and point to the supporting run IDs, configurations, logs, checkpoints, and evaluation outputs. Full configuration and per-seed details belong in Appendices A and B.

| Model | Planned seeds | Completed seeds | Run IDs | Config/log/checkpoint evidence | Status or exclusion reason |
|---|---|---|---|---|---|
| Linear | TODO | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | TODO | TODO |
| CNN | TODO | TODO | TODO | TODO | TODO |
| RNN (cell: TODO) | TODO | TODO | TODO | TODO | TODO |
| Transformer | TODO | TODO | TODO | TODO | TODO |

### 3.2 Main Quantitative Comparison

Report selected-checkpoint test metrics, the test sample count, and the aggregation method (for example, mean ? standard deviation across run seeds). Put per-seed values in Appendix B.

| Model | Test accuracy (mean ? SD) | Test macro-F1 (mean ? SD) | Test loss (mean ? SD) | Parameters (total / trainable) | Best epoch(s) |
|---|---:|---:|---:|---:|---:|
| Linear | TODO | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | TODO | TODO |
| CNN | TODO | TODO | TODO | TODO | TODO |
| RNN (cell: TODO) | TODO | TODO | TODO | TODO | TODO |
| Transformer | TODO | TODO | TODO | TODO | TODO |

TODO: Summarize the ranking and practical size of differences. Do not claim statistical significance without supporting analysis.

### 3.3 Learning Dynamics and Convergence

Describe convergence, stability across seeds, train/validation gaps, selected checkpoint epochs, and stopping reasons from the logged histories. Discuss underfitting or overfitting only when supported by the curves and metrics.

**Figure 3.1 ? Training and validation loss.** Use comparable axes where meaningful and show seed variability if available.

![TODO: Verified training and validation loss curves](figures/results/TODO_loss_curves.png)

**Figure 3.2 ? Training and validation metrics.** Plot accuracy and macro-F1 over epochs; use separate panels if needed for readability.

![TODO: Verified training and validation accuracy and macro-F1 curves](figures/results/TODO_metric_curves.png)

| Model | Epochs completed | Best-checkpoint epoch(s) | Stopping reason | Learning-dynamics observation |
|---|---:|---:|---|---|
| Linear | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | TODO |
| CNN | TODO | TODO | TODO | TODO |
| RNN | TODO | TODO | TODO | TODO |
| Transformer | TODO | TODO | TODO | TODO |

### 3.4 Efficiency and Accuracy--Compute Trade-offs

State the timing hardware/software and protocol. Distinguish forward-only inference timing from training wall-clock time.

| Model | Training time (min; mean ? SD) | Device | Precision | Batch size | Median batch latency (ms) | Throughput (images/s) | Amortized latency (ms/image) |
|---|---:|---|---|---:|---:|---:|---:|
| Linear | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| CNN | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| RNN | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Transformer | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

TODO: Interpret the accuracy/efficiency trade-off, noting any differences in timing conditions or tuning budgets.

### 3.5 Confusion Matrices and Class-Level Results

Use a consistent class order: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. State whether matrix cells show counts or row-normalized proportions. Put additional matrices in Appendix C.

![TODO: Main test-set confusion matrix or compact multi-model comparison, with consistent labels and scale](figures/results/TODO_main_confusion_matrices.png)

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| TODO | T-shirt/top | TODO | TODO | TODO | TODO |
| TODO | Trouser | TODO | TODO | TODO | TODO |
| TODO | Pullover | TODO | TODO | TODO | TODO |
| TODO | Dress | TODO | TODO | TODO | TODO |
| TODO | Coat | TODO | TODO | TODO | TODO |
| TODO | Sandal | TODO | TODO | TODO | TODO |
| TODO | Shirt | TODO | TODO | TODO | TODO |
| TODO | Sneaker | TODO | TODO | TODO | TODO |
| TODO | Bag | TODO | TODO | TODO | TODO |
| TODO | Ankle boot | TODO | TODO | TODO | TODO |

TODO: Discuss the main class-level patterns and cite supporting matrices or metrics.

### 3.6 Qualitative Predictions and Error Analysis

Show representative correct and incorrect predictions with stable sample IDs, true/predicted labels, and (optionally) predicted probabilities. Identify the evaluation split and ground explanations in the images and model outputs.

![TODO: Labeled panel of correct and incorrect predictions](figures/results/TODO_prediction_examples.png)

| Sample ID | True class | Predicted class | Model | Predicted-class probability | Observation |
|---|---|---|---|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO |

### 3.7 Representation and Inductive-Bias Discussion

Relate observed behavior to flattened pixels (linear/MLP), spatial feature maps (CNN), ordered sequences (RNN), and image tokens (Transformer). Treat architectural explanations as interpretations, and consider optimization, capacity, and tuning as alternatives.

| Result observation | Supporting evidence | Possible representation/inductive-bias explanation | Caveat or alternative explanation |
|---|---|---|---|
| TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO |

### 3.8 Ablation / Extension Results (Optional)

Keep optional experiments separate from the main comparison. Identify what changed, fixed controls, and whether test results remained blind to selection.

| Ablation / extension | Changed factor | Fixed controls | Validation result | Test result (if test-blind) | Interpretation |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

### 3.9 Implementation Issues and Verified Corrections

Document issues encountered and corrections verified by tests or experiment evidence; mark verification as pending otherwise.

| Issue or observed failure | Correction | Verification evidence | Impact on reported results |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

### 3.10 Limitations and Threats to Validity

Discuss seed count/variance, tuning budget, hardware variability, model-specific optimization, split/preprocessing choices, dataset scope, and incomplete or failed runs. Distinguish measured limitations from potential threats.

| Limitation / threat | Evidence or scope | Likely effect on interpretation | Mitigation or remaining work |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

### 3.11 Conclusion

Answer the research questions from Section 1 using the evidence above. State the strongest measured model, its compute/size trade-off, cross-seed consistency, and the key qualification. Keep conclusions specific to this dataset and protocol.

TODO: Write the evidence-based conclusion after the main results are complete.

## Reproducibility and Artifact Access

List only existing, checked artifacts. Provide a code revision and enough run/configuration detail for another person to reproduce the reported metrics.

| Artifact | Path or URL | Revision / run ID | Verification note |
|---|---|---|---|
| Code revision | TODO | TODO | TODO |
| Effective configurations | TODO | TODO | TODO |
| Run logs and histories | TODO | TODO | TODO |
| Checkpoints and evaluation outputs | TODO | TODO | TODO |
| Figures and prediction records | TODO | TODO | TODO |
| Slides / presentation video | TODO | TODO | TODO |

## Member Contributions and AI Usage Disclosure

| Member | Implementation | Experiments / verification | Writing / editing | Presentation |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |

Initial repository/interface assistance: OpenAI Codex, 2026-09-04, exact model identifier not recorded; human verification pending. Extend this disclosure with actual AI usage, affected sections, and verification performed; keep it consistent with `AI_USAGE.md`.

## References

TODO: List verified primary sources actually consulted. Ensure in-text citations and bibliography entries correspond.

## Appendix A: Full Configurations and Architecture Details

Provide the effective configuration for each main run (or link to a checked artifact): model architecture, optimizer, learning rate, scheduler, batch size, epoch cap, early-stopping settings, seed, split, and preprocessing.

| Model | Run/config ID | Architecture settings | Optimization settings | Data/preprocessing settings | Effective config artifact |
|---|---|---|---|---|---|
| Linear | TODO | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | TODO | TODO |
| CNN | TODO | TODO | TODO | TODO | TODO |
| RNN | TODO | TODO | TODO | TODO | TODO |
| Transformer | TODO | TODO | TODO | TODO | TODO |

## Appendix B: Per-Seed Results and Run Provenance

| Model | Run ID | Seed | Best epoch | Test accuracy | Test macro-F1 | Test loss | Training time (min) | Git revision / dirty state |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

TODO: Include split and normalization hashes, environment details, and run exclusions or deviations.

## Appendix C: Additional Curves, Confusion Matrices and Error Examples

Place full-size per-model plots and matrices here when they are too dense for the main text. Keep labels, normalization, and class ordering consistent.

![TODO: Additional per-model training curves](figures/results/TODO_appendix_curves.png)

![TODO: Linear confusion matrix](figures/results/TODO_linear_confusion_matrix.png)

![TODO: MLP confusion matrix](figures/results/TODO_mlp_confusion_matrix.png)

![TODO: CNN confusion matrix](figures/results/TODO_cnn_confusion_matrix.png)

![TODO: RNN confusion matrix](figures/results/TODO_rnn_confusion_matrix.png)

![TODO: Transformer confusion matrix](figures/results/TODO_transformer_confusion_matrix.png)

![TODO: Additional labeled error examples](figures/results/TODO_appendix_error_examples.png)

## Appendix D: Detailed Verification and Reproduction Notes

TODO: Provide exact commands and environment details needed to reproduce runs and evaluations, verification/test evidence, artifact checksums where applicable, and notes on known deviations. Do not label a command verified unless it ran successfully.
