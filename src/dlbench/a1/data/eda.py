"""Owner A (Thiên): descriptive data analysis, not test-based model tuning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def generate_eda(config: Mapping[str, Any], output_dir: Path) -> None:
    """Export counts, class distribution, shape/range checks and representative images.

    Include imbalance and leakage checks, plus dataset source/license.
    Raw preparation outputs go to runs; curated images go to docs/assets/a1.
    """
    raise NotImplementedError("TODO A (Thiên): EDA tables/plots from actual dataset observations.")
