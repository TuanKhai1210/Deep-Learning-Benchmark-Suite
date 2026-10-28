"""Apply final project settings and retain portable reproduction evidence."""
from pathlib import Path
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
main=ROOT/'main.tex'
source=main.read_text(encoding='utf-8')
source=source.replace(r'\newcommand{\TODO}{\par\noindent\textbf{TODO}\par}'+'\n','')
source=source.replace(r'\TODO',r'\input{sections/00-abstract}')
source=source.replace('Nguyen Hao Thiên','Nguyen Hao Thien')
source=source.replace('Shared training pipeline, metrics, checkpoints, and timing',
                      'Shared engine, MLP/GRU, metrics, checkpoints, and timing')
source=source.replace('Dataset, EDA, preprocessing, and visual error analysis',
                      'Dataset, EDA, preprocessing, Linear/CNN, and error analysis')
source=source.replace('Configuration, reproducibility, system integration, and documentation',
                      'Transformer, configurations, artifacts, integration, and report')
extra=r"""
\usepackage[section]{placeins}
\usepackage{needspace}
\usepackage{xurl}
\makeatletter
\renewcommand\small{\@setfontsize\small{11pt}{13.2pt}}
\renewcommand\footnotesize{\@setfontsize\footnotesize{10pt}{12pt}}
\makeatother
\captionsetup{font=small,labelfont=bf}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible}
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{12pt plus 2pt minus 2pt}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\renewcommand{\topfraction}{.95}
\renewcommand{\bottomfraction}{.90}
\renewcommand{\textfraction}{.05}
\hypersetup{pdftitle={Tiki-taka A1 Final Report: Architecture Comparison on Fashion-MNIST},
pdfauthor={Nguyen Anh Khoa; Nguyen Hao Thien; Ta Tuan Khai}}
"""
if r'\usepackage{xurl}' not in source:
    source=source.replace(r'\begin{document}',extra+'\n'+r'\begin{document}')
if r'\newcolumntype{Y}' not in source:
    source=source.replace(r'\usepackage{xurl}',r'\usepackage{xurl}'+'\n'+r'\newcolumntype{Y}{>{\raggedright\arraybackslash}X}')
source=source.replace(r'{|c|X|c|X|c|}',r'{|c|>{\raggedright\arraybackslash}X|c|>{\raggedright\arraybackslash}X|c|}')
main.write_text(source,encoding='utf-8')
(ROOT/'sections/00-abstract.tex').write_text(r"""
Across 15 Fashion-MNIST training runs, the custom Transformer achieves the highest
mean test accuracy and macro-F1, 91.267\% and 91.234\%, narrowly ahead of the
custom CNN at 90.690\% and 90.726\%.

The comparison uses five architecture families, one persisted stratified split and
three seeds per configuration. Selected checkpoints are evaluated on the official
10,000-image test source, with mean and sample standard deviation reported across seeds.

The CNN offers close predictive performance with about 72.9\% fewer parameters
than the Transformer and lower observed training-loop cost. Separately measured
Windows CPU inference favors different trade-offs; the model choice depends on
quality, size and the intended runtime.

All classifiers use shared preprocessing and metrics, while their representations
range from flattened pixels to spatial feature maps, image rows and patch tokens.
The study reports validation/test metrics, exact parameter counts, learning curves,
T4 train-and-validation loop times, explicitly labeled CPU timing, confusion
matrices and selected successes/failures. Shirt and related upper-body categories
remain difficult across models. Small seed count, different training policies and
separate timing environments prevent a causal or universal ranking of model families.
Effective configurations, run/checkpoint identities, verification records and
reconstruction instructions accompany the editable report source.
""".strip()+'\n',encoding='utf-8')
copies={
 PROJECT/'analysis/a1-postprocess-2026-10-24/a1_colab_test_analysis.ipynb':ROOT/'reproduction/a1_colab_test_analysis.ipynb',
 PROJECT/'analysis/a1-timing-only-2026-10-24/a1_colab_timing_only.ipynb':ROOT/'reproduction/a1_colab_timing_only.ipynb',
 PROJECT/'analysis/a1-cpu-results-2026-10-24/verification.json':ROOT/'evidence/original_evaluation_verification.json',
 PROJECT/'analysis/a1-timing-only-2026-10-24/validation.json':ROOT/'evidence/original_timing_validation.json',
}
for src,dst in copies.items():
    shutil.copyfile(src,dst)
(ROOT/'evidence/AI_USAGE_report.md').write_text("""# Report and analysis AI assistance

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
""",encoding='utf-8')
print('Finalized main source and portable evidence')
