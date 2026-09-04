# Environment records — owner C (Khải)

Choose a compatible Python/PyTorch/torchvision/CUDA installation, verify it on
the team's machine, then record the exact dependency versions and installation
instructions. The package declares dependencies in `setup.py`; an experimentally
validated ML environment has not yet been frozen.

Record OS, Python, torch/torchvision, CUDA/driver, GPU model/VRAM, CPU and precision.
Keep separate records if training hardware differs. Re-measure comparable timing
on the same agreed benchmark machine. Never store credentials or full environment
variables in these files.

Upstream references:

- [Setuptools project configuration](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
