# PPoPP'26 MetaAttention Artifact Evaluation

## 0. Overview

This repository contains the artifacts for the PPoPP'26 Artifact Evaluation of paper #238: "MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends".


### Badges Claimed

- Artifacts Available

  + All code related to MetaAttention is publicly available in this repository.
- Artifacts Evaluated – Functional

  + We provide detailed documentation and Docker support to build, install, and test MetaAttention. The examples/ folder contains all Attention operators discussed in the paper, and testing/ contains functional correctness tests.
- Results Reproduced
  + We provide Docker images and scripts to reproduce the main performance results presented in the paper (Figure 11 and Figure 14).

### Claims Supported by Artifact

- Functionality: MetaAttention can automatically generate correct kernels for diverse attention variants (e.g., RetNet, Mamba2, MLA) across different backends.
- Performance: MetaAttention achieves performance comparable to hand-optimized libraries for previously supported attention mechanisms, and significantly outperforms native implementations (e.g., PyTorch) for mechanisms where no hand-optimized library exists on H100 and MI250 GPUs.


# 1. Getting Started Guide

This section guides you through setting up the environment and running a example to verify functionality.

## Hardware Requirements

To reproduce the results in the paper, specific hardware is required:

- 1x NVIDIA Hopper GPU

, or

- 1x AMD MI200 Series GPU


## Prepare Docker Environment

To ease the process of installing all the dependencies, baseline software, and MetaAttention code, we provide a Dockerfile and a simple guideline to build a Docker image with all of above installed.

### For NVIDIA GPU
```bash
# clone the repo or use the archive we provided
git clone https://github.com/smallscientist1/AttentionEngineFork.git AttentionEngine --recursive -b ppopp_AE
cd AttentionEngine/docker
# take about 50 minutes
docker build -t metaattn_cuda -f Dockerfile.cu128 .

docker run -it --gpus all --name metaattn-AE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/..:/AttentionEngine metaattn_cuda

cd /AttentionEngine
```

### For AMD GPU
```bash
git clone https://github.com/smallscientist1/AttentionEngineFork.git AttentionEngine --recursive -b ppopp_AE
cd AttentionEngine/docker
# take about 80 minites on a 32-core machine
docker build -t metaattn_rocm -f Dockerfile.rocm .

docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd)/..:/AttentionEngine \
  metaattn_rocm

cd /AttentionEngine
```

## Running Example

Inside the Docker container, run the following script to execute the RetNet Attention example (corresponding to Fig 5 and Fig 7 in Section 3 of the paper). This verifies that the installation is successful and the system can generate/run kernels.
```bash
# expect < 2 minutes
python examples/retention_parallel.py 
```
**Expected Output**: The script should print AttentionEngine Succuessfully created.

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

