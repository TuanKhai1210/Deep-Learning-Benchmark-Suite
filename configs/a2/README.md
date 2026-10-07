# Assignment 2 configuration status

The team proposes **TrashCan 1.0 Material semantic segmentation**.
See the [proposal and evidence notes](../../reports/a2/README.md) and
[project page](../../docs/a2.md).

Instructor approval is pending. This directory does not yet contain an executable
training configuration, and main experiments must wait for approval.

The proposal specifies split seed `36`, a video-disjoint 5,057/1,077/1,078 image
partition, 17 output classes and ignore ID `255`. Initial run seed is `69420`;
paired repeats `67` and `69` are optional, compute-dependent plans.
The full manifest remains in the working EDA bundle at
`eda_v1/splits/trashcan_material_seed36_v1.json`; consult the proposal notes
for its content hash and working-folder link.

After approval, record the frozen converter, category map, normalization,
optimization, pilot-confirmed budget, metrics and checkpoint rule in versioned
configs. Reuse genuinely generic helpers from `dlbench.common`; do not assume
the A1 classification engine or metrics implement semantic segmentation.
