# Assignment 2 configuration status

The team proposes **TrashCan 1.0 Material semantic segmentation**.
See the [proposal and evidence notes](../../reports/a2/README.md) and
[project page](../../docs/a2.md).

Initial code is available in this feature branch (25 October 2026):
[implementation and commands](../../docs/a2_initial_code.md) and
[notebook](../../notebooks/a2_initial_code.ipynb). Instructor approval remains
unconfirmed; main experiments require approval and a pilot-confirmed frozen protocol.

`protocol.py` holds shared executable **draft** defaults. `unet.py` (U),
`segformer_off.py` (S0), and `segformer_on.py` (S1) are literal experiment configs.
S0/S1 differ only in run name and train-only color jitter. AdamW, constant learning
rate, loss weights and budget are initial development choices, not tuned results.

The proposal specifies split seed `36`, a video-disjoint 5,057/1,077/1,078 image
partition, 17 output classes and ignore ID `255`. Initial run seed is `69420`;
paired repeats `67` and `69` are optional, compute-dependent plans.
The full manifest remains in the working EDA bundle at
`eda_v1/splits/trashcan_material_seed36_v1.json`; consult the proposal notes
for its content hash and working-folder link.

After approval, use the pilot to review and freeze optimization/budget in a
versioned protocol and record approval/pilot evidence references. The CLI rejects
main training while these fields are unresolved. A2 uses its own segmentation
engine, loss and metrics; generic seeding/environment helpers come from common.
