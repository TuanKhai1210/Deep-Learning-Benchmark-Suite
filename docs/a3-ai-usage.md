---
layout: default
title: "Assignment 3 AI Usage Disclosure"
---
# Assignment 3 — AI Usage Disclosure

## Scope

This disclosure covers generative AI assistance used during the **M1 Dataset Proposal Milestone** for Assignment 3 (Multimodal Product Type Classification on Amazon Berkeley Objects). AI tools served strictly as assistive programming, formatting, and documentation resources. In accordance with course guidelines, main multimodal experiments and benchmark measurements have not yet been executed and will commence only upon official proposal approval.

## Review and Verification Protocol

- **Code Review:** All generated data ingestion scripts, candidate filtering routines, feature cache extraction pipelines, and fusion head architectures were inspected and reviewed by team members.
- **Data Integrity:** Dataset statistics, candidate counts (7,221 pairs across 10 classes), and the product-group disjoint splitting rules were verified against raw catalog metadata from `abo-listings.tar` and `images.csv.gz`.
- **Feasibility Pilot:** The 300-pair feasibility pilot was executed on local CPU hardware; tensor shapes ($B \times 128 \to B \times 384$) and exact parameter counts (214,666 parameters for Product and Mean heads) were verified directly through PyTorch debuggers and unit tests.
- **Academic Authorship:** The research design, title masking policy, controlled ablation hypotheses, and written proposal interpretations remain the intellectual product and responsibility of the student authors.

## Specific Use Areas

1. **Dataset Ingestion & Candidate Filtering:**
   - *Tool:* Google Gemini (3.8 Flash).
   - *Contributor:* Nguyễn Hạo Thiên.
   - *Scope:* Assisted in structuring JSON parsing heuristics for Amazon.com listings, checking image index coverage, and drafting product-group disjoint splitting algorithms.
   - *Human Verification:* Verified that zero duplicate images or product variants leak across partitions, and confirmed all 10 product categories are covered in each split.

2. **Feature Caching Pipeline & Lexical Masking:**
   - *Tool:* OpenAI Codex (v5.6 Sol).
   - *Contributor:* Nguyễn Anh Khoa.
   - *Scope:* Assisted in drafting DistilBERT title tokenization routines, class phrase masking logic, and ResNet-18 ImageNet-1K V1 feature extraction caching tools.
   - *Human Verification:* Validated that masked titles correctly replace class strings with `[MASK]` tokens, and confirmed that NPZ feature caches match manifest sample IDs.

3. **Multimodal Fusion Architectures & Proposal Documentation:**
   - *Tool:* OpenAI Codex (v5.6 Sol / GPT-6).
   - *Contributor:* Tạ Tuấn Khải.
   - *Scope:* Assisted in authoring the PyTorch modules for Concat, Product fusion, and parameter-matched Mean ablation, structuring the 300-pair feasibility pilot, and formatting the LaTeX proposal document.
   - *Human Verification:* Verified that Product fusion ($[v, t, v \odot t]$) and Mean ablation ($[v, t, (v+t)/2]$) possess identical parameter counts (214,666 parameters) for a fair causal comparison.

## Academic Responsibility

The authors (Nguyễn Hạo Thiên, Nguyễn Anh Khoa, Tạ Tuấn Khải) assume full academic responsibility for the integrity, correctness, and originality of all submitted proposal materials. No synthetic experimental data or unrun benchmark numbers are claimed as completed findings.

The complete chronological audit log is maintained in the repository root document: [AI_USAGE.md](../AI_USAGE.md).
