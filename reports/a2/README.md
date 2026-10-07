# Assignment 2 — TrashCan-Material proposal

Published in the repository on **7 October 2026**. Group: **Tiki-taka**.

## Proposal

[Open the proposal PDF](Tiki-taka_A2_TrashCan_Material_Proposal.pdf).
The file is an unchanged copy of the team-supplied `trashcan_material_proposal (1).pdf`; it contains a proposal, not completed training results.

- [Project page](https://tuankhai1210.github.io/Deep-Learning-Benchmark-Suite/a2.html)
- [Group working data and EDA folder](https://drive.google.com/drive/folders/1VwT0UyleGYyF9yrCtXaOwZpnp-ZRrfGD)
- [Official dataset source](https://irvlab.cs.umn.edu/resources/trashcan)

Instructor approval and LMS submission have **not** been confirmed. Publishing on GitHub/Pages is not LMS submission. Main experiments require approval.

## Measured evidence

The small JSON summaries below were measured on 5 October 2026; they are copied unchanged from the previously verified local EDA bundle.

- [Metadata and original-split findings](evidence/metadata.json): dataset identity, annotation hashes, class support, original source partitions and video overlap.
- [New split summary](evidence/split_summary.json): seed-36 train/validation/test class and pixel support.
- [Verification record](evidence/verification.json): 7,212 unique images, 312 video groups, zero pairwise group overlap and class-coverage checks.

The **full** manifest remains in the accompanying EDA bundle on the group working folder:
`eda_v1/splits/trashcan_material_seed36_v1.json`.
Canonical content SHA-256 (excluding the digest field itself):
`87905f1a97dd2c04194e049ca61d1a7f3bbe2942e9fe17dc2d737c371cdce943`.
The digest is not a PDF checksum. The full manifest is not bundled in this repository publication; the verification record describes the full EDA artifact, not a repository-local manifest.

| Partition | Images | Video groups | Annotations |
|---|---:|---:|---:|
| Train | 5,057 | 207 | 8,563 |
| Validation | 1,077 | 54 | 1,956 |
| Test | 1,078 | 51 | 1,817 |

The original train/validation partitions share **127 filename-derived video groups**. Retain the original notebook and EDA outputs in `evidence/eda_original/` within the working evidence folder without overwriting them. Use `eda_v1/` and its new manifest for future training. This repository update does not move, overwrite or upload those Drive files.

## Proposed protocol

- TrashCan 1.0 Material; 16 foreground IDs + background 0; ignore 255.
- Native polygon rasterization: Pillow 12.3.0, same-class union, cross-class overlap ignored. Reuse the same converter for training targets and evaluation ground truth.
- Aspect-ratio-preserving 512 × 512 letterbox; padding ignored.
- U-Net scratch vs pretrained SegFormer-B0; color-augmentation on/off controlled within SegFormer.
- Primary: foreground macro mIoU; also foreground Dice, trash-only mIoU and per-class scores.
- Best checkpoint: validation foreground mIoU, then lower loss, then earlier epoch.
- Initial run seed 69420; optional paired repeats 67 and 69. Split seed 36 is separate.
- Budget and optimization settings are provisional until the post-approval pilot.

## Attribution and data handling

Dataset authors: **Jungseok Hong, Michael Fulton and Junaed Sattar**. Source imagery: **JAMSTEC**. Retain the supplied `LICENSE.txt` and the COCO JSON license records; the proposal describes both. Commercial permission and redistribution conditions must not be inferred from academic access.

Raw frames, the dataset archive, large checkpoints and credentials are not included in this commit. The JSON files contain derived counts, provenance hashes and video-group identifiers, not raw images or polygon annotations.
