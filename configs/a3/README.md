# A3 Amazon Berkeley Objects

ABO product-type classification uses one product image and one English listing title.
The pipeline compares five classifier heads on shared frozen features. The configuration
in [head.py](head.py) uses the proposal budget of eight epochs, batch size 128,
learning rate 0.001 and a 128-dimensional projection.

## Verified benchmark

The official run is `abo-v1-main`. See the [benchmark record](benchmark-results.md)
for held-out results, inputs, fingerprints, and instructions for replaying the run
with the team artifacts.

## Team workflow

Run the notebooks in order using a common workspace and run ID.

| Notebook | Owner and output |
|---|---|
| [01 Prepare](../../notebooks/a3_01_prepare.ipynb) | Thiên audits and reviews pairs, duplicates and variants. Khoa reviews train-derived mask additions. Outputs the group-disjoint manifest, QA receipt and two caches. |
| [02 Train](../../notebooks/a3_02_train.ipynb) | Trains image-only, title-only, concat, product and mean-ablation for seeds 69420, 67 and 69. Saves checkpoint and validation for each run. |
| [03 Test](../../notebooks/a3_03_test.ipynb) | After validation review, scores the locked checkpoints on the shared held-out test split. |
| [04 Analysis](../../notebooks/a3_04_analysis.ipynb) | Rechecks checkpoints, predictions and membership. Exports score tables, matched errors, per-class F1, confusion matrices and learning curves. |

### Divide training across team members

After Notebook 01, one coordinator uses Notebook 02's **lock** mode to write
`benchmark_lock.json` in the cohort folder. Every participant uses the same
frozen cohort, the original two NPZ cache files, and the same source bundle.
The lock checks the manifest digest, both cache fingerprints, effective head
settings, config-file fingerprint and A3/common source fingerprints.

In **batch** mode, give each assignment a distinct BATCH_ID. Set MODELS and
SEEDS to run their Cartesian product in a single execution. Set PAIRS instead
to list exact assignments such as `[("image-only", 67), ("concat", 69)]`.
Rerun the same batch ID and assignment after interruption to reuse verified
completed runs. Batch folders can be copied to the coordinator under
`WORK/batches`. Select **collect** and list their IDs to assemble the complete
five-model, three-seed validation suite. The collector reports missing or
duplicate pairs and never chooses between duplicate results. Test and analysis
operate on the assembled suite. Analysis retains per-run fit times and
hardware details rather than pooling training times across participants.

Notebook 01 retrieves the pinned listings/index and only the selected ABO small
images. Run through the audit first, edit the decisions CSV in UTF-8, then rerun
the freeze cell with a reviewer name. The audit queues 10 readable pairs per
class, near-image pairs, brand/model and title-stem matches, and exact-pixel
label conflicts.
Exact RGB pixel duplicates share a group. Reviewers can add manual exclusions
and links. For a two-ID case, merge joins the groups, separate requires different groups,
and keep retains both samples without adding either constraint. The frozen
split is versioned and records exclusions and class
counts. Set ENCODE to True after selecting any train-derived extra phrases.
The classifier heads use the same two caches. Unimodal variants open only their
required cache.

The suite locks the stronger of image-only and title-only by mean validation
macro-F1 across seeds. An exact tie selects image-only. Test reports the
paired per-seed differences for concat versus that reference, product versus
concat, and product versus mean-ablation. Analysis computes sample SD across
seeds and uses a common color scale for row-normalized confusion matrices.
Timing records encoder extraction, train plus validation epochs, and held-out
scoring with cached vectors. The last measurement includes batching, transfer,
loss and metrics and can be reported alongside the encoder costs.

For a shared handoff keep the cohort and decisions, image inventory, two caches,
the source ZIP, and the complete run folder. Analysis needs the run folder.
Images are optional for displaying error examples. Cite the
[official ABO data release](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html)
when reproducing the dataset or sharing examples.

## Install and run a synthetic check

From the A3 checkout, install matching Torch and Torchvision builds for the target
device and run:

```sh
python -m pip install -e ".[a3]"
dlbench-a3 smoke --output runs/a3/smoke-01
```

You can also use `python -m dlbench.a3.cli` in place of `dlbench-a3`.
The smoke command uses random vectors with all ten classes in each partition and
executes train, checkpoint selection and test for all five configurations.

## Prepare one dataset manifest

Get `abo-listings.tar` and `abo-images-small.tar` from the
[official ABO release](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html).
Use `images/metadata/images.csv.gz` from the release and the
extracted `images/small/` directory. Keep dataset files below the ignored
`data/abo/` directory.

```sh
dlbench-a3 candidates --listings data/abo/abo-listings.tar --image-index data/abo/images/metadata/images.csv.gz --output data/abo/candidates.csv
```

Thiên checks image decoding and image–title pairing, removes unusable rows,
and assigns the same `group_id` to confirmed duplicates and product variants
in `data/abo/reviewed.csv`. Candidate group IDs start as image IDs.
Membership before pixel QA is provisional.

| CSV field | Meaning |
|---|---|
| `sample_id` | Unique listing ID |
| `image_id` | Unique primary image ID |
| `image_path` | Relative path under the small-image root |
| `title` | Original en_US listing title |
| `label` | Product type from the proposed ten-class list |
| `group_id` | Product group, including reviewed duplicates and variants |
| `split` | train, validation or test, added by the split command |

```sh
dlbench-a3 split --reviewed data/abo/reviewed.csv --output data/abo/manifest.csv
dlbench-a3 validate-manifest --manifest data/abo/manifest.csv
```

The deterministic splitter keeps groups intact and aims for 70/15/15 per class
using split seed 36. Check the printed class counts before freezing the manifest.
A class missing from any split is an error. Label indices follow the ordered
`CLASSES` tuple in `dlbench.a3.data`.

## Extract features and train

```sh
dlbench-a3 encode --modality image --manifest data/abo/manifest.csv --image-root data/abo/images/small --output data/abo/image.npz
dlbench-a3 encode --modality title --manifest data/abo/manifest.csv --output data/abo/title.npz
dlbench-a3 train --model concat --manifest data/abo/manifest.csv --image-cache data/abo/image.npz --title-cache data/abo/title.npz --config configs/a3/head.py --seed 69420 --run-id concat-69420
dlbench-a3 test --manifest data/abo/manifest.csv --image-cache data/abo/image.npz --title-cache data/abo/title.npz --run-dir runs/a3/concat-69420
```

Encoders download weights on their first use. Images use frozen ResNet-18
ImageNet-1K V1 with its 224-pixel transforms. Titles use the pinned
DistilBERT revision and a 64-token cap. Title caches mask every class phrase by
default. Khoa can add train-derived variations with `--extra-mask-phrases`.
Use a separate cache with `--raw-titles` for the raw-title comparison.

An NPZ contains Unicode IDs, FP32 features, and JSON metadata with the modality,
encoder identity and manifest digest. Caches are joined by ID and must cover
the full manifest. `--device cpu` forces CPU. Use `image-only`
with the image cache, `title-only` with the title cache, and both caches
for `concat`, `product` and `mean-ablation`.
The last two have identical head shapes and differ in the elementwise product
versus the elementwise mean of the projected vectors.

Training writes run provenance, history, and the best checkpoint chosen by
validation macro-F1, with the earliest epoch winning ties. Use a new run ID to
restart a failed run. Run the test command after the validation protocol is
locked. It verifies checkpoint, manifest and cache identity and writes held-out
metrics and predictions by ID in a new test directory. Epoch timing covers cached-head training and
validation. Encoder timing is recorded in cache metadata.

## Run the matched fusion experiments

Use the same reviewed manifest, encoder caches, head budget and device across
concat, product fusion and mean ablation. Choose the seed list before training.
The following example uses the three project run seeds.

```sh
dlbench-a3 train-fusion --manifest data/abo/manifest.csv --image-cache data/abo/image.npz --title-cache data/abo/title.npz --config configs/a3/head.py --seeds 69420 67 69 --output runs/a3/fusion-v1
dlbench-a3 test-fusion --manifest data/abo/manifest.csv --image-cache data/abo/image.npz --title-cache data/abo/title.npz --output runs/a3/fusion-v1
```

Training writes a copy of the manifest, the individual runs and a suite
index with input and training artifact digests. Start a failed training suite
in a fresh output directory. Review validation results before running the test
command. The test command checks all training artifacts before scoring,
reuses completed test directories on the requested device and recomputes
metrics from the predictions before writing `comparison.json`. The comparison
contains the per-seed macro-F1 and accuracy, plus paired macro-F1 differences
for product versus concat and product versus the matched mean ablation.
A single seed has no sample standard deviation.

For the team handoff, retain the encoder NPZ files alongside the suite output.
Keep images, caches and model checkpoints in the experiment storage.
