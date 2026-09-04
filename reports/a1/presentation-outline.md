# A1 Presentation Outline

Status: presentation planning. Confirm the course's presentation/video constraints before assigning durations.

| Segment | Content to prepare | Proposed speaker |
|---|---|---|
| Context | Problem, dataset and comparison questions | Thiên (A) |
| Data | EDA, saved split, preprocessing/augmentation and leakage controls | Thiên (A) |
| Common pipeline | Training, metrics, checkpoint selection and reproducibility | Khoa (B) |
| Models | Explain each representation and key design choices | Each model's owner |
| Results | Main table, curves, timing method and compute trade-offs | Khoa (B), with input from all |
| Analysis | Confusion patterns, example errors, representation/inductive bias | Thiên (A) and Khải (C) |
| Limitations and conclusion | What evidence supports, what remains uncertain | Khải (C) |
| Disclosure and reproducibility | Member contributions, AI usage and how to rerun | Khải (C), with confirmation from all |

## Readiness checklist

- [ ] All members appear/present as required and can defend their contributions.
- [ ] Every number comes from reviewed run evidence.
- [ ] Slides distinguish test results from validation-based selection.
- [ ] Units, hardware, batch size, repetitions and seed aggregation are visible where relevant.
- [ ] Figures are readable and use consistent class labels/model names.
- [ ] Conclusions are proportional to the dataset and experiment budget.
- [ ] References and AI disclosure are included.
- [ ] Final report, slides, video, repository and checkpoint/reconstruction links work.

Rehearse explanations of leakage prevention, macro-F1, the checkpoint rule, each model's representation, and why timing comparisons are meaningful. Do not present TODO tables or unimplemented code as completed work.
