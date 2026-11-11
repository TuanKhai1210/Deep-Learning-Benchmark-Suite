---
layout: default
title: "Assignment 2 AI Usage Disclosure"
---
# Assignment 2 — AI Usage Disclosure

## Scope

This disclosure covers AI tool assistance during the preparation, implementation, verification, and documentation of Assignment 2 (Video-Disjoint Semantic Segmentation on TrashCan-Material). AI tools served strictly as assistive programming, formatting, and analysis utilities; all experimental data, training runs, checkpoint weights, and evaluation metrics were produced on actual physical GPU accelerators (Tesla T4) without synthetic fabrication.

## Review and Verification Protocol

- **Code Review:** All generated model architectures (U-Net, SegFormer-B0 adaptation), loss operations, letterbox adapters, and metric accumulators were inspected, reviewed, and validated by team members.
- **Data Integrity:** Dataset ingestion, polygon-to-mask rasterization, and the seed-36 video-disjoint manifest were independently audited with automated scripts and visual sanity checks (`annotation_qa.png`).
- **Metric Verification:** All reported test metrics, per-class IoU/Dice scores, confusion matrices, and paired differences were recomputed from raw integer confusion counts using the standalone verification script (`reports/a2_reporting/reproduction/verify_results.py`).
- **Artifact Traceability:** Every cited run is bound to its configuration snapshot, training history, cryptographic checkpoint hash, and execution log in `reports/a2_reporting/verified_results.json`.

## Specific Use Areas

1. **Dataset Ingestion & Split Construction:**
   - *Tool:* Google Gemini (3.8 Flash).
   - *Contributor:* Nguyễn Hạo Thiên.
   - *Scope:* Assisted in exploring video sequence ID parsing heuristics from filename strings to eliminate temporal leakage across frames.
   - *Human Verification:* Verified 100% video disjointness ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$) using set-intersection assertions over all 312 video groups.

2. **Loss Formulation & Segmentation Engine:**
   - *Tool:* OpenAI Codex (v5.6 Sol).
   - *Contributor:* Nguyễn Anh Khoa.
   - *Scope:* Assisted in drafting PyTorch tensor operations for combined Cross-Entropy and batch-level soft Dice loss, numerical stability epsilon ($\epsilon = 10^{-6}$), and ignore-mask handling.
   - *Human Verification:* Tested with synthetic mini-batches containing zero-foreground edge cases to confirm correct loss masking without division-by-zero or gradient explosion.

3. **Model Adaptation, Reproduction Tooling & LaTeX Formatting:**
   - *Tool:* OpenAI Codex (v5.6 Sol / GPT-6).
   - *Contributor:* Tạ Tuấn Khải.
   - *Scope:* Assisted in structuring the SegFormer-B0 classification head replacement, letterbox preprocessing, 20-step GPU pilot harness, LaTeX table code generation from CSV metrics, and standalone reproduction scripts.
   - *Human Verification:* Validated tensor dimensions ($B \times 17 \times 512 \times 512$), confirmed that valid pixel sums in confusion matrices match partition totals ($186,339,960$ test pixels), and verified checkpoint hashes against model binaries.

## Academic Responsibility

The authors (Nguyễn Anh Khoa, Nguyễn Hạo Thiên, Tạ Tuấn Khải) assume full academic responsibility for the integrity, correctness, and originality of all submitted materials. Generative AI was not used to fabricate numerical data, alter predictions, or simulate training runs.

The complete chronological audit log is maintained in the repository root document: [AI_USAGE.md](../AI_USAGE.md).
