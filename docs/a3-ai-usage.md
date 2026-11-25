---
layout: default
title: "Assignment 3 AI Usage Disclosure"
---
# Assignment 3 — AI Usage Disclosure

## Scope

This disclosure covers generative AI assistance used across the **M1 Dataset Proposal** and **M2 Progress Draft** milestones of Assignment 3 (Multimodal Product Type Classification on Amazon Berkeley Objects). AI tools served strictly as assistive programming, formatting, audit verification, and report drafting resources. All quantitative benchmark figures originate from physical GPU execution logs, deterministic evaluation checkpoints, and verified run artifacts.

## Review and Verification Protocol

- **Code Review:** All generated data ingestion scripts, candidate filtering routines, feature cache extraction pipelines, head architectures, training loops, and evaluation suites were inspected and validated by team members.
- **Data Integrity & Audit:** Dataset statistics, candidate exclusion decisions (5 listings dropped out of 7,221: 1 German title `B0777Z915W`, 4 out-of-scope items `B0759Z13XN`, `B07B4M93BC`, `B0778X4L9J`, `B07F2G6417`), and product-group disjoint splitting rules (5,281 groups across 7,216 listings) were verified against raw catalog metadata and image caches.
- **Experimental Verification:** All nine head training runs (Image-only, Title-only, and Concat across seeds 69420, 67, 69) were executed on Tesla T4 GPUs. All reported validation/test Macro-F1 and accuracy scores were programmatically cross-checked against serialized JSON result artifacts and checkpoint weights.
- **Academic Authorship:** The research formulation, hypothesis testing, title masking policy, qualitative error taxonomy, and empirical interpretations remain the sole intellectual responsibility of the student authors.

## Specific Use Areas

1. **Dataset Ingestion, Review Audit & Visual Modality:**
   - *Tool:* Google Gemini (3.8 Flash).
   - *Contributor:* Nguyễn Hạo Thiên.
   - *Scope:* Assisted in JSON parsing heuristics for raw Amazon.com listings, identifying candidate catalog categories, and auditing photo aspect ratios / center crop retention.
   - *Human Verification:* Manually verified image decoding across candidate pairs, reviewed candidate exclusions (99.93% acceptance rate), and confirmed zero product group overlap across train, validation, and test splits.

2. **Feature Caching Pipeline & Title Masking Protocol:**
   - *Tool:* OpenAI ChatGPT (GPT-4o) &amp; OpenAI Codex.
   - *Contributor:* Nguyễn Anh Khoa.
   - *Scope:* Assisted in crafting regular expressions for canonical category string masking, verifying DistilBERT WordPiece sequence length caps, and structuring the evaluation suite.
   - *Human Verification:* Audited masking impact across all 7,216 titles (3,334 titles modified; 46.2%), verified that maximum subword length was 53 tokens (well within the 64-token cap), and confirmed checkpoint selection logic.

3. **Multimodal Fusion Modeling, Suite Orchestration & M2 Report Drafting:**
   - *Tool:* OpenAI Codex (v5.6 Sol / GPT-6).
   - *Contributor:* Tạ Tuấn Khải.
   - *Scope:* Assisted in authoring PyTorch modules for unimodal heads and Concat fusion, automating multi-seed training scripts, generating LaTeX tables, and drafting narrative sections of the M2 report.
   - *Human Verification:* Verified exact parameter counts (83,466 for Image-only, 116,234 for Title-only, 198,282 for Concat), confirmed identical optimizer budgets, and personally inspected and analyzed matched error cases (`B075X4WBLM`, `B001BQAMNW`, `B075YMY2J7`).

## Academic Responsibility

The authors (Nguyễn Hạo Thiên, Nguyễn Anh Khoa, Tạ Tuấn Khải) assume full academic responsibility for the integrity, correctness, and originality of all submitted materials. No synthetic experimental data or unrun benchmark numbers are claimed as completed findings.

The complete chronological audit log is maintained in the repository root document: [AI_USAGE.md](../AI_USAGE.md).
