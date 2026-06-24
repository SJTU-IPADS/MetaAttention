# PPoPP'26 MetaAttention Artifact Evaluation

## 0. Overview

This repository contains the artifacts for the PPoPP'26 Artifact Evaluation of paper #238: "MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends".

MetaAttention provides users with pythonic interface to define customized attention flexibly and automatically generate device code with high performance. Examples of various attention mechanisms (e.g., softmax attention, RetNet, Mamba2, MLA) are provided in the examples/ folder.

# 1. Getting Started Guide

This section covers the supported quickstart paths on the `feat/quickstart` branch.

## Hardware Requirements

To run generated kernels, you still need a supported GPU runtime:

- 1x NVIDIA Hopper GPU, or
- 1x AMD MI200 Series GPU

`uv pip install -e .` now works on this branch. The editable install exposes the project packages (`attn_engine`, `core`, `autotuner`, `benchmark`) directly, so local development no longer depends on setting `PYTHONPATH` by hand.

## Local Quickstart with `uv`

Use this path if you already have a working Python, PyTorch, and GPU driver stack on the host machine.

### NVIDIA / CUDA

```bash
uv venv --python 3.12
. .venv/bin/activate
uv pip install -e .
```

This installs MetaAttention itself and resolves the pinned `torch` and `tilelang` sources from `pyproject.toml`.

### Verify the installation
```bash
# run from the repo root
python examples/retention_parallel.py
```

Expected output includes `AttentionEngine Succuessfully created.`

Notes:

- The example needs a working GPU runtime. On this workstation, the script fails before kernel generation if no NVIDIA driver is present.
- The editable install was validated by importing `attn_engine`, `core`, `autotuner`, and `benchmark` after `uv pip install -e .`.

## Docker Quickstart

Use Docker if you want a reproducible environment with the heavyweight dependencies preinstalled.

### NVIDIA GPU

```bash
# takes about 50 minutes
docker build -t metaattn_cuda -f docker/Dockerfile.cu128 .

docker run -it --gpus all --name metaattn-cuda \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  metaattn_cuda
```

### AMD GPU

```bash
# takes about 80 minutes on a 32-core machine
docker build -t metaattn_rocm -f docker/Dockerfile.rocm .

docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  metaattn_rocm
```

Inside either container, the repository is already installed in editable mode. Run:
```bash
cd /workspace/MetaAttention
python examples/retention_parallel.py
```

Expected output includes `AttentionEngine Succuessfully created.`

# 2. Step-by-Step Instructions

# Functional tests

Run the following command to verify the correctness of various supported Attention operators mentioned in Sections 3 & 4 of the paper.
```bash
# may take about 10 minutes
python testing/test.py
```
This script runs tests for parallel and recurrent patterns against reference implementations (e.g., PyTorch).

**Expected Output**: The script should print All tests passed. (There may be some warnings from baselines, but they can be ignored.)


# Performance tests

We consider Figure 11 and Figure 14 to be the key results of our paper, demonstrating the performance of MetaAttention-generated operators on hardware.
The following are the steps to replicate these experiments.

**Note for reproducing the result**:
Baseline libraries (e.g., FlashAttention, FlashLinearAttention) are frequently updated, thus slight variations in performance numbers compared to the static plots in the paper are expected. However, the overall conclusion should remain consistent: MetaAttention achieves performance comparable to hand-written libraries and better than native PyTorch.


## Figure 11

- Target Device: NVIDIA H100-80GB GPU
- Description: Evaluates the performance of MetaAttention against baselines on the H100.

Run the following command:
```bash
# take about 90 minutes
python testing/benchmark_h100.py
```
The figure will be saved as `./figure11_h100.pdf`.

## Figure 14

- Target Device: AMD MI250X GPU
- Description: Evaluates the performance of MetaAttention on the AMD backend.

Run the following command:
```bash
# take about 20 minutes
python testing/benchmark_mi250.py
```
The figure will be saved as `./figure14_mi250.pdf`.
