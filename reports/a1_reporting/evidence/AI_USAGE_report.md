# Report and analysis AI assistance

- Tool: OpenAI Codex; exact serving model IDs for individual sessions were not retained.
- User and responsible integration member: Ta Tuan Khai.
- Period: 23-26 October 2026, Asia/Bangkok.
- Purpose: handbook-based report structure, LaTeX source completion, run/test artifact verification,
  CPU-capable timing-only tooling, figure/table generation and technical copy editing.
- Prompt summaries: prepare one-pass test/analysis cells; explain missing CPU timing; review the current PDF;
  fix the supplied LaTeX and complete the report from actual results.
- Affected material: all report sections, generated tables/figures, evidence summaries and reconstruction instructions.
- AI contribution: proposed and wrote technical explanations and helper code, recomputed saved-result summaries,
  identified inconsistent claims, assembled the editable source and compiled/rendered the PDF.
- Verification: original handbook; pinned source a8539be; actual configs/history/checkpoint hashes;
  supplied validation/test predictions; original timing session and raw batches; canonical raw IDX MD5 checks;
  official Fashion-MNIST/PyTorch references; numeric, reference and visual PDF checks.
- Student responsibility: module owners remain responsible for their implementations and interpretation;
  Ta Tuan Khai is responsible for integration and final verification. No unrecorded human sign-off is asserted.
- Measured results originate from saved artifacts. Report generation performs no training or checkpoint inference.
  The displayed preprocessing batch is regenerated data, not a logged training batch.

Historical tool/member/model labels and representative prompts are retained separately in AI_USAGE_source.md.
