# Contributing

This project is maintained by Nguyễn Hạo Thiên (A), Nguyễn Anh Khoa (B), and Tạ Tuấn Khải (C). Development is currently focused on Assignment 1.

## Workflow

1. Start a focused feature branch from the current `main`.
2. Keep changes within the agreed module contract; discuss shared-interface changes before implementing them.
3. Add or update tests, configuration, and documentation alongside code.
4. Run the relevant checks and report their actual outcome, including skipped tests and known limitations.
5. Open a pull request and request review from another member.
6. Merge after review; do not commit data, credentials, local logs, or model weights.

Suggested branch names include `feat/a1-data-eda`, `feat/a1-engine`, `feat/a1-transformer`, and `docs/a1-results`.

## Ownership and review

| Owner | Primary areas | Reviewer |
|---|---|---|
| Thiên (A) | Data, EDA, preprocessing, Linear, CNN | Khoa (B) |
| Khoa (B) | Engine, trainer, metrics, checkpoint, timing, MLP, RNN | Khải (C) |
| Khải (C) | Config, seeds, logging, integration/tests, Transformer, documentation integration | Thiên (A) |

Ownership coordinates implementation, not exclusive access. Each member writes tests and technical notes for their own contribution. See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for the file map.

## Before requesting review

- Run `python -m unittest discover -s tests -v` in the project environment.
- Run `python -m dlbench.a1.cli validate-config --config configs/a1/models/linear.py` and validate any other changed model config.
- Inspect the diff for unrelated changes, secrets, large artifacts, broken links, and unsupported claims.
- Keep source configs as literal Python `CONFIG` dictionaries with no imports or side effects.
- Update the experiment contract when changing shared rules. Do not mark the protocol frozen while required choices are unresolved.
- Preserve split seed `36`, run seeds `[69420, 67, 69]`, and one common saved split unless the team explicitly approves a versioned change.
- Record AI assistance and verification in `AI_USAGE.md` when applicable.

Passing development tests does not imply ML training is complete. Tests for pending implementations must remain visibly pending until implemented and verified.

## Experimental evidence

Every reported metric must be traceable to the effective configuration, split, seed, checkpoint, environment and Git revision. Never tune with the official test set, fabricate measurements, or choose only the most favorable seed. Publish small reviewed figures/tables under `docs/assets/a1/`; document access or reconstruction for larger artifacts.
