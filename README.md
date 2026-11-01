# MetaAttention

> The PPoPP'26 Artifact Evaluation is available on the [`PPoPP_AE` branch](https://github.com/SJTU-IPADS/MetaAttention/tree/PPoPP_AE).

## 0. Overview

MetaAttention provides users with pythonic interface to define customized attention flexibly and automatically generate device code with high performance. Examples of various attention mechanisms (e.g., softmax attention, RetNet, Mamba2, MLA) are provided in the examples/ folder.

The [Stateful example guide](examples/README.md) documents Delta Rule, KDA,
GDN2, ComBa, IPLR/DPLR, Gated Oja, fixed-step Delta Product, and ABC, including
input conventions, state continuation, numerical limits, and validation.
The [domain glossary](GLOSSARY.md) defines State, Transition, Readout and coverage
terms. Schema v5 adds [typed Scan Stage DAGs](docs/API.md#scan-stage-dag-schema-v5),
Preconditioned KDA/GDN and stable LightNet, including shared auxiliary heads,
state continuation, gradients and optional checkpoint.

# 1. Getting Started Guide

## Hardware Requirements

To run generated kernels, you still need a supported GPU runtime:

- 1x NVIDIA Hopper GPU, or
- 1x AMD MI200 Series GPU

The dedicated Gated Delta Rule engine additionally supports NVIDIA H20. Its
first-release contract is fixed-length, head-first BF16 training with FP32
gate/beta/state, 128-dimensional key/value heads, and 64-token chunk alignment.

## Local Quickstart with `uv`

Use this path if you already have a working Python environment and a PyTorch build that matches your target runtime (CUDA, ROCm, or CPU) on the host machine.

### Install
```bash
uv venv --python 3.12
. .venv/bin/activate
# install a platform-appropriate PyTorch build first
uv pip install torch
# install the pinned TileLang release with its runtime dependencies
uv pip install "tilelang==0.1.15"
# then install this project without re-resolving torch or TileLang
uv pip install --no-deps -e .
```

This installs MetaAttention itself from the local checkout. PyTorch is intentionally host-selected so CUDA, ROCm, and CPU environments can provide their own compatible build, and `--no-deps` avoids replacing that choice during editable install.
The project pins the released TileLang `0.1.15` API in `pyproject.toml` and `uv.lock`. Specialized Stateful lowering families use TileLang `@tilelang.jit`/`@T.prim_func` kernels, including diagonal summaries/carries/readouts, low-rank/WY/checkpoint and backward paths, scalar and rank-one scans, generated expression loaders, and packed execution.

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

# imports for official example factories
uv run --extra test pytest -q -m examples tests/unit/test_example_imports.py
```

Run the GPU correctness suite explicitly:
```bash
uv run --extra test pytest -q --run-gpu -m 'functional and gpu' tests/functional
```
This executes the legacy 10-case parity matrix plus direct-engine and official v2 factory coverage. Forward results, backward gradients where supported, reference implementations, dtypes, and operator-specific tolerances are checked. Without `--run-gpu`, GPU tests are collected but skipped with an explicit reason; without a CUDA/HIP-visible device, the hardware fixture skips them.

Run the H20 GDN functional suite on an otherwise idle H20:
```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra test pytest -q --run-gpu \
  -m 'functional and h20' tests/functional/test_gdn_h20.py
```
This executes the generic Stateful Operator Gated Delta Rule profile for
single- and multi-chunk execution, initial/final State, custom scale, and all
supported gradients. See `examples/gated_delta_rule.py` and `docs/API.md` for
the public interface.

Stateful Operator correctness and performance diagnostics use
`testing/benchmark_stateful_plans.py`; its independent authored-order recurrence
is the correctness oracle for output-only, final-State, and continuation contracts.
### Compare Stateful Operator with FLA

Install the FLA 0.5.2 release in its optional uv dependency group:

```bash
uv sync --group fla-benchmark
```

Run the conservative first-phase matrix on an otherwise idle NVIDIA GPU:

```bash
uv run --group fla-benchmark python testing/benchmark_fla_stateful.py \
  --families linear,simple-gla,gla,normalized-linear,retention,rwkv6,gsa,hgrn \
  --contract all --fla-baseline both --timing both --workspace-gib 8 \
  --output-dir benchmark_results/stateful_fla
```

The runner selects a physical GPU before importing CUDA and uses the installed
FLA 0.5.2 release by default. To compare a local checkout explicitly, pass
`--fla-source ../flash-linear-attention`. The manifest records the actual FLA
import path, package versions and checkout revision when supplied.
`--fla-baseline both` (the default) compares recurrent and chunk/parallel prefill
APIs in separate records; select `recurrent` or `prefill` to run one schedule.
Missing public APIs are recorded as unavailable, without schedule fallback.
JSON/CSV record `fla_baseline` and `fla_function`, and heatmaps separate schedules.
`--workspace-gib` sets the Meta compilation budget (default: 1 GiB); 8 GiB
allows the larger matrix points on an otherwise idle H20 with sufficient free
memory. This is a budget, not a reservation. The runner includes public-call
preparation in both sides, validates MetaAttention against its independent authored-order
recurrence, validates directly comparable FLA outputs, and only then reports
`FLA / MetaAttention` speedups. Families without a semantic FLA mapping, or
contracts for which FLA does not expose the same State, are recorded as
non-comparable rather than included in the ratio. The legacy
`testing/benchmark_flash_attention.py` entry point now forwards to this runner;
it no longer compares the unrelated softmax FlashAttention package. The JSON,
CSV, and plots are artifacts; benchmark results require a fixed GPU, software
environment, warmup/repetition count, and idle-device record.
The standalone Stateful plan runner measures those contracts without external baseline adapters.

The main FLA runner now defaults to all 25 registered families, including
`preconditioned-kda`, `preconditioned-gdn` and `lightnet`. These staged operators
use the same batch/head matrix, state contracts, timing and result files as the
other families. See the [unified stage benchmark guide](docs/stateful-stage-benchmarks.md#unified-benchmark-interface)
for examples, selected child runtimes and FLA semantic limits. The focused
`testing/benchmark_fla_stages.py` additionally measures VJPs and supplied
nonzero caches under its separately documented timing contract.

The [new example benchmark guide](examples/README.md#benchmark-against-fla)
covers Delta Rule, KDA, GDN2, ComBa, IPLR/DPLR, Oja, Delta Product and ABC.



## Performance tests

For performance on previously unseen Stateful Operator compositions, see the
[expression generalization suite](docs/stateful-generalization-tests.md). It
scans seeded input expressions, feature gates, ordered rank-one propagations,
multiple writes/readouts and their combinations, verifies outputs and final
State, and reports latency, memory and the stable complexity envelope.

We consider Figure 11 and Figure 14 to be the key results of our paper, demonstrating the performance of MetaAttention-generated operators on hardware.
The following are the steps to replicate these experiments.



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
