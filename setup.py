"""Package metadata. Experiment settings live in configs/a1/*.py."""

from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="deep-learning-benchmark-suite",
    version="0.1.0",
    description="Reproducible deep learning architecture comparisons for CO3133.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/TuanKhai1210/deep-learning-benchmark-suite",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],
    # Dependency declarations, not a validated/frozen ML environment.
    # Select compatible torch/torchvision builds for the team's hardware.
    extras_require={
        "ml": ["torch", "torchvision", "numpy", "scikit-learn", "matplotlib"],
        "notebooks": ["jupyterlab"],
    },
    entry_points={"console_scripts": ["dlbench-a1=dlbench.a1.cli:main"]},
)
