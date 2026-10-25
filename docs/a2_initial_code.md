# A2 initial code — TrashCan-Material

Milestone: **initial implementation**, 25 October 2026. These files provide a
reviewable pipeline; they do not establish dataset integration or main results.
See [proposal/evidence](../reports/a2/README.md) and [configs](../configs/a2/README.md).

## Included

| Component | Implementation / contract |
|---|---|
| Data | `src/dlbench/a2/data.py`: existing seed-36 manifest, original source annotation hashes, category map, group disjointness and per-image native target hash checks |
| Converter | `converter.py`: four function bodies ported unchanged from the 5 October EDA tools; Pillow 12.3.0, same-class union/cross-class ignore |
| Preprocessing | RGB bilinear/mask nearest, centered 512 letterbox, target padding 255, RGB black padding, fixed ImageNet mean/std for both models |
| Augmentation | Train-only color jitter p=0.5 before letterbox/normalization; factors 0.9–1.1, hue ±0.01, random operation order; stateless seed/epoch/image RNG |
| Models | Four-downsampling-stage U-Net (base 16, GroupNorm, bilinear upsampling); ADE20K SegFormer-B0, fresh 17-class classifier, raw logits resized to target |
| Objective | CE + foreground soft Dice, equal initial weights; target-present foreground classes per batch; ignore removed from numerator and denominator |
| Metrics | Full-partition 17×17 confusion including background; foreground mIoU/Dice, trash mIoU (9–16), all-class mIoU/Dice, per-class support/predicted, ignore coverage |
| Selection | Best validation foreground mIoU, then lower validation loss; exact ties keep earlier epoch; patience 8 based on this comparator |
| Artifacts | Effective config, source snapshots/hashes, Git revision/status/diff, package/hardware info, initial state hash, history, metrics, best/last checkpoints |
| Resume | Epoch boundary only; model/AdamW/RNG/best/patience/history; same code/config/device/environment/data membership required |

The manifest uses IDs 0–16; there is **no ADE20K label reduction**. For metrics,
classes with zero union are null and excluded from macro means; a prediction-only
class has IoU zero and is included. Defined-class counts accompany macro scores.
Epoch loss is a valid-pixel-weighted mean of batch CE+Dice objectives; it is not
a single whole-partition soft-Dice calculation. All-ignore batches skip optimizer
updates; an epoch with no supervised pixels fails explicitly.

Parameter counts are explicit: smoke outputs contain `total_params`,
`trainable_params` and `frozen_params`; each new run records the same fields
under `metadata.json` → `parameter_counts`. Counts include unique parameter
elements, exclude buffers and use `requires_grad` to distinguish trainable/frozen.
Current U-Net and SegFormer fine-tune all parameters (frozen = 0); pretrained
initialization does not freeze the backbone.

SegFormer source is
[`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512),
pinned revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da` (resolved 25 October 2026).
The wrapper follows the official
[Transformers SegFormer API](https://huggingface.co/docs/transformers/v4.57.6/en/model_doc/segformer).
Download failures are errors; they never silently switch to random weights.
Random initialization is available explicitly for synthetic development checks.

## Install and check without raw data

From the repository/worktree root, choose a compatible torch/torchvision build
for the hardware, then install the A2 extra in a virtual environment:

```console
python -m pip install -e ".[a2]"
python -m dlbench.a2.cli validate-config --config configs/a2/unet.py
python -m dlbench.a2.cli validate-config --config configs/a2/segformer_off.py
python -m dlbench.a2.cli validate-config --config configs/a2/segformer_on.py
python -m dlbench.a2.cli smoke --config configs/a2/unet.py
python -m dlbench.a2.cli smoke --config configs/a2/segformer_off.py
python -m unittest discover -s tests -v
```

`smoke` performs one synthetic optimization step, batch 2 × 64×64, CPU by default.
It does not inspect real data or demonstrate model accuracy. Add `--pretrained`
to explicitly exercise pinned SegFormer weights. `validate-config --main` fails
while draft/approval/pilot fields are pending, as intended.

Dependencies pin Pillow 12.3.0 and Transformers 4.57.6. Torch/torchvision remain
hardware-selected; metadata captures their actual versions. This is a development
dependency declaration, not a validated Colab environment lock.

## External data layout

```text
TrashCanMaterial/
  instances_train_trashcan.json
  instances_val_trashcan.json
  train/vid_......_frame.......jpg
  val/vid_......_frame.......jpg
eda_v1/splits/trashcan_material_seed36_v1.json
```

Use the previously published manifest from the group's EDA folder. Do not rebuild
the split per model or move raw images into the repository. The original source
partition in each row remains the lookup path after the new video-disjoint split.
The manifest is external to Git; the portable project package also includes a
byte-identical convenience copy (no frames or annotations).

```console
python -m dlbench.a2.cli check-manifest --manifest /path/to/trashcan_material_seed36_v1.json
python -m dlbench.a2.cli check-data --config configs/a2/unet.py --data-root /path/to/TrashCanMaterial --manifest /path/to/trashcan_material_seed36_v1.json --limit 4
```

`check-data` verifies all source annotation hashes/membership, then loads and
checks native masks for the selected first images. It does not claim to read
every RGB image. Raw image content hashes were not part of the EDA manifest;
dimensions and semantic target hashes are checked. Inspect overlays on real
images before a pilot. Each model uses the same adapter and converter.

## Explicit small development integration (when real data is ready)

```console
python -m dlbench.a2.cli train --config configs/a2/unet.py --data-root /path/to/TrashCanMaterial --manifest /path/to/trashcan_material_seed36_v1.json --output runs/a2/U-dev-001 --development --limit 8 --epochs 1
python -m dlbench.a2.cli train --config configs/a2/unet.py --data-root /path/to/TrashCanMaterial --manifest /path/to/trashcan_material_seed36_v1.json --output runs/a2/U-dev-001 --development --limit 8 --epochs 2 --resume
python -m dlbench.a2.cli evaluate --run-dir runs/a2/U-dev-001 --data-root /path/to/TrashCanMaterial --manifest /path/to/trashcan_material_seed36_v1.json --split validation --output runs/a2/U-dev-001/validation_replay.json
```

Development limits slice train/validation only. Standalone validation replay
uses the full validation split. `last.pt` is authoritative for resume and can
repair interrupted derived history/metrics writes. New output directories must
not exist. Run one process per output directory. Checkpoints use PyTorch pickle;
load only trusted files. Mid-epoch resume, concurrent writers and identical
results across devices or package versions are outside this initial contract.

The notebook [a2_initial_code.ipynb](../notebooks/a2_initial_code.ipynb) calls these
same modules. It supports an existing checkout or upload of the portable source
ZIP in Colab; default execution validates configs and runs synthetic checks.
Real-data checks and development training require explicit paths/flags. Main
training is not enabled by the notebook.

## Paired experiment and next gate

U vs S0 compares entire systems (architecture and pretraining both differ).
S0 vs S1 uses equal seed 69420, initial model state, sampler, geometry, optimizer,
loss and maximum budget. Color randomness does not consume global model/sampler
RNG. Seeds 67/69 remain optional paired repeats, not initial requirements.
`comparison.color_decision` validates paired config/source/environment/data/init
identity and evaluation provenance before applying the validation-only rule:
trash mIoU improves ≥0.01, foreground mIoU/Dice each drop ≤0.01. It rejects test
metrics for augmentation selection. Official test evaluation rejects development
checkpoints. Freeze choices before test use.

Remaining gates: teammate review, real-image batch/overlay checks, real-data
forward/backward, tiny-subset overfit, approval evidence, GPU pilot and frozen
optimization/budget. Current defaults are AdamW (U lr 1e-3, S0/S1 lr 6e-5), weight
decay .01, no scheduler/AMP, batch 4, max 40 epochs, patience 8. These are
provisional implementation choices. No full training, GPU capacity measurement,
submission or test score is claimed.
