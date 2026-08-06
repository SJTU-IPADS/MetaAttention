# MetaAttention

> The PPoPP'26 Artifact Evaluation is available on the [`PPoPP_AE` branch](https://github.com/SJTU-IPADS/MetaAttention/tree/PPoPP_AE).

## 0. Overview

MetaAttention provides users with pythonic interface to define customized attention flexibly and automatically generate device code with high performance. Examples of various attention mechanisms (e.g., softmax attention, RetNet, Mamba2, MLA) are provided in the examples/ folder.

# 1. Getting Started Guide

## Hardware Requirements

To run generated kernels, you still need a supported GPU runtime:

- 1x NVIDIA Hopper GPU, or
- 1x AMD MI200 Series GPU

## Local Quickstart with `uv`

Use this path if you already have a working Python environment and a PyTorch build that matches your target runtime (CUDA, ROCm, or CPU) on the host machine.

### Install

```bash
uv venv --python 3.12
. .venv/bin/activate
# install a platform-appropriate PyTorch build first
uv pip install torch
# then install this project without re-resolving torch
uv pip install --no-deps -e .
```

This installs MetaAttention itself from the local checkout. PyTorch is intentionally host-selected so CUDA, ROCm, and CPU environments can provide their own compatible build, and `--no-deps` avoids replacing that choice during editable install.

### Verify the installation
```bash
# run from the repo root
python examples/retention_parallel.py
```

Expected output includes `AttentionEngine Succuessfully created.`

# 2. Step-by-Step Instructions

## Functional tests

Run the unified pytest suite from the repository root. The unit suite is CPU-safe and checks imports, code generation, benchmark dispatch, CSV schemas, and PDF plotting without compiling GPU kernels:
```bash
uv run --extra test pytest -q tests/unit

# imports for all 14 official example factories
uv run --extra test pytest -q -m examples tests/unit/test_example_imports.py
```

Run the GPU correctness suite explicitly:
```bash
uv run --extra test pytest -q --run-gpu -m 'functional and gpu' tests/functional
```
This executes the legacy 10-case parity matrix plus direct-engine and official v2 factory coverage. Forward results, backward gradients where supported, reference implementations, dtypes, and operator-specific tolerances are checked. Without `--run-gpu`, GPU tests are collected but skipped with an explicit reason; without a CUDA/HIP-visible device, the hardware fixture skips them.

**Expected output**: all unit tests pass. On a supported GPU, all functional tests run without unconditional skips. Optional baseline warnings do not replace MetaAttention correctness checks.


## Performance tests

We consider Figure 11 and Figure 14 to be the key results of our paper, demonstrating the performance of MetaAttention-generated operators on hardware.
The following are the steps to replicate these experiments.

**Note for reproducing the result**:
Baseline libraries (e.g., FlashAttention, FlashLinearAttention) are frequently updated, thus slight variations in performance numbers compared to the static plots in the paper are expected. However, the overall conclusion should remain consistent: MetaAttention achieves performance comparable to hand-written libraries and better than native PyTorch.


## Figure 11

- Target Device: NVIDIA H100-80GB GPU
- Description: Evaluates the performance of MetaAttention against baselines on the H100.

Run the guarded long-run pytest entrypoint (about 90 minutes on the target hardware):
```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra test --extra bench pytest -q \
  --run-benchmarks -m 'benchmark and h100' tests/benchmarks/test_h100.py
```
The runner executes all 62 Figure 11 configurations in an isolated pytest temporary directory, validates 24 CSV files containing `MetaAttention` rows, and renders a non-empty `figure11_h100.pdf`. Missing optional baselines are reported and omitted rather than replaced with fabricated values.

## Figure 14

- Target Device: AMD MI250X GPU
- Description: Evaluates the performance of MetaAttention on the AMD backend.

Run the guarded long-run pytest entrypoint (about 20 minutes on the target hardware):
```bash
HIP_VISIBLE_DEVICES=0 uv run --extra test --extra bench pytest -q \
  --run-benchmarks -m 'benchmark and mi250' tests/benchmarks/test_mi250.py
```
The runner executes all 25 Figure 14 configurations in an isolated pytest temporary directory, validates 10 CSV files containing `MetaAttention` rows, and renders a non-empty `figure14_mi250.pdf`. On a non-HIP runtime or non-MI250 device, it skips with the unmet hardware prerequisite.
