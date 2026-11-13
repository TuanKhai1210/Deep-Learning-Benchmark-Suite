# A3 Amazon Berkeley Objects base code

ABO product-type classification uses one product image and one English listing title.
The base provides five configurations over cached frozen features. The shared head
budget in [head.py](head.py) is a starting point for the pilot.

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
