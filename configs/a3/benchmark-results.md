# ABO benchmark results

Run `abo-v1-main` uses 7,216 Amazon Berkeley Objects image–title pairs in ten
catalog classes. The split keeps the 5,281 reviewed groups intact and contains
5,052 training, 1,083 validation and 1,081 test listings. The input cohort and
caches were verified on 21 November 2026. The run comparison and analysis were
verified on 22 November 2026.

| Model | Validation macro-F1 (%) | Test macro-F1 (%) | Test accuracy (%) |
| --- | ---: | ---: | ---: |
| Image-only | 91.39 ± 0.33 | 90.69 ± 0.39 | 92.01 ± 0.23 |
| Title-only | 86.85 ± 0.92 | 87.73 ± 1.11 | 88.87 ± 0.81 |
| Concat | 94.92 ± 0.18 | 94.38 ± 0.40 | 94.73 ± 0.32 |
| Product | 95.05 ± 0.29 | 95.21 ± 0.85 | 95.25 ± 0.90 |
| Mean ablation | 94.74 ± 0.41 | 94.20 ± 0.11 | 94.63 ± 0.24 |

Scores are mean ± sample standard deviation over seeds 69420, 67 and 69 on one
fixed split. Image-only is the strongest unimodal model on validation. On test,
concat gains 3.69 percentage points over image-only. Product gains 0.83 points
over concat and 1.01 points over the equal-size mean ablation. Both product
comparisons have a positive paired macro-F1 difference for all three seeds.

All heads train for eight epochs using frozen ResNet-18 and DistilBERT vectors.
The head fit timings include validation on a Tesla T4. The reported held-out
scoring times cover the CPU scoring loop on cached vectors. The encoder passes
were measured separately on a Windows CPU at batch size 32.

## Reproduce the recorded run

Use the [exact source ZIP](abo-v1-source.zip) and the original `cohorts/abo-v1`
folder from the team workspace. Provide the reviewed manifest and freeze receipt,
`benchmark_lock.json`, and both original NPZ caches. To re-score the original
checkpoints or regenerate analysis, use the team's complete `runs/abo-v1-main`
folder. If training new checkpoints, use a distinct run ID. Training on a
different device or software build can change artifacts.

Run [Prepare](../../notebooks/a3_01_prepare.ipynb) to check the cohort, then use
[Train](../../notebooks/a3_02_train.ipynb) for batch training and collection.
[Test](../../notebooks/a3_03_test.ipynb) scores the collected validation
checkpoints. [Analysis](../../notebooks/a3_04_analysis.ipynb) verifies predictions
and exports the result tables. See the [run guide](README.md#team-workflow) for
the workspace layout and notebook inputs. Rebuilding the same curated cohort
from the public ABO release also requires the team's review decisions. The
source archive preserves the byte-level source identity expected by the lock.

| Artifact | SHA-256 |
| --- | --- |
| Manifest identity recorded by the pipeline | `9742854841e0b0d1afe752dc080a990707b47ba82a109a0180b5abb90a158d1a` |
| `image.npz` | `b6747e111149dfe0b68c0b883a5151a6278e8a37d0847f4af391730be8cabbbb` |
| `title.npz` | `ab68abf2968d0eea57a8a946655bb125480507c224caa76db8711ce27a5e9c52` |
| `benchmark_lock.json` | `2d72b043ddaade015dbfd883b6d1b253e4881191b4e40aa2ab66d5c7be8e7af3` |
| `abo-v1-source.zip` | `f0f61c8b41bbc418517149067c4f7c6ceae39516f5c634841da86651694640a4` |
| `suite.json` | `fcf7478cccba89818e45719f85cf004d978f26090ae6a3479bbebf6b7ef9f215` |
| `comparison.json` | `efd6de5d1382ddd7837a2f53a84621c1cc89198c7e7afeb40a7a1ac90e001838` |

Dataset source and license appear on the [official ABO release page](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html).
