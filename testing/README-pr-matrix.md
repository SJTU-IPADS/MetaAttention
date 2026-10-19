# PR validation matrix runner

Run in the SSH workspace `allinai2plus:/home/linyanxuan`, with the project at
`/home/linyanxuan/linyanxuan/MetaAttentionImpl`, using `.venv/bin/python`.
`run_pr_matrix.py` is a standard-library controller for the existing benchmark
entry points. It does not change their numerical or timing contracts.

The controller prepends the active interpreter's bin directory to child `PATH`,
so `.venv/bin/python` works without shell activation and nested extension builds
can find `.venv/bin/ninja`. On this host the system toolkit is CUDA 13 while
PyTorch uses CUDA 12.8. Install the matching development headers before running
the legacy C++/CUDA extension tests (these packages are already installed here):

```bash
uv pip install --python .venv/bin/python \
  'nvidia-cuda-cccl-cu12==12.8.90' 'nvidia-cuda-nvcc-cu12==12.8.93'
```

The existing extension templates prefer these Python CUDA headers over the
incompatible system CUDA 13 headers.

## Inspect the full plan without executing tests

```bash
cd /home/linyanxuan/linyanxuan/MetaAttentionImpl
.venv/bin/python -m testing.run_pr_matrix plan \
  --output-dir benchmark_results/pr_matrix_plan
```

The full profile contains 900 jobs:

| Stage | Jobs | Expected benchmark records |
| --- | ---: | ---: |
| CPU tests | 1 | — |
| GPU tests (functional files and GPU-marked unit tests) | 27 | — |
| Hardware benchmark artifact tests | 2 | — |
| FLA comparison | 750 | 6,000 |
| Expression generalization | 54 | 972 |
| Same-IR optimized/generic compositions | 6 | 24 |
| Stage forward/VJP | 48 | 48 workloads, up to five backends each |
| Additional plan diagnostics | 12 | 36 |

FLA: all 25 families, B=1/2, H=1/8/16, T=63/64/65/130/2048/8192,
three contracts, two schedules; T=1 separately covers two contracts. BF16 and
family-specific dimensions are inherited from the existing runner. Workspace is
1 GiB. Warmup=10 and repetitions=30, with public CUDA, wall and Graph timing.

Generalization: six families, every level 0..8, seeds 1701/2028, T=65/512/2048,
and B/H/K/V shapes 1/4/64/64, 2/8/64/64, 1/4/128/128. Warmup=3,
10 paired batches of five calls. Default slowdown/jitter thresholds are retained.

Compositions: four kinds, B=1/2, T=65/512/2048, 10 paired batches. The legacy
composition runner does not check an independent oracle during measurement;
records are marked `NOT_CHECKED_BY_RUNNER`. Its correctness coverage is supplied
by the preceding test phase and must not be described as an in-run oracle check.

Stages: three families, K=V=64/128, H=1/8, T=1/64/2048/8192, BF16,
1 GiB workspace, 30 repetitions. Short-sequence VJPs use the existing gradient
oracle; long-sequence VJPs retain `MEASURED_UNVALIDATED`. Stage external-layout
prebinding differs from the FLA public interval; compare within each runner.
Plan diagnostics cover mamba2 and pressure, six lengths, all three contracts,
using their factory dimensions and default 1 GiB Target.

## Smoke validation

CPU smoke (no GPU work):

```bash
.venv/bin/python -m testing.run_pr_matrix smoke --cpu-only \
  --output-dir benchmark_results/pr_matrix_smoke_cpu
```

This runs selected benchmark/controller tests, all five entry-point `--help`
commands, and a generalization dry-run. GPU-marked tests remain skipped.

GPU smoke (run when a GPU is idle):

```bash
.venv/bin/python -m testing.run_pr_matrix smoke \
  --gpu 0 --output-dir benchmark_results/pr_matrix_smoke_gpu
```

This adds one GPU correctness test, six linear/FLA comparisons, two expression
points, four compositions, one LightNet forward/VJP workload and three linear
plan contracts. Small repetition counts make these exploratory validation only.
Specify `--gpu N` to require physical GPU N; omission selects an idle GPU.
GPU smoke returning zero confirms collection, not that every backend is valid;
inspect `report.md` and raw correctness statuses before the full run.

## Full execution

Only start this command after checking GPU smoke results:

```bash
cd /home/linyanxuan/linyanxuan/MetaAttentionImpl
mkdir -p benchmark_results/pr_matrix_full
nohup .venv/bin/python -u -m testing.run_pr_matrix full --gpu 0 \
  --fla-source /mnt/nvme/linyanxuan/flash-linear-attention \
  --output-dir benchmark_results/pr_matrix_full \
  > benchmark_results/pr_matrix_full/controller.log 2>&1 < /dev/null &
```

The controller selects an idle physical GPU when it first needs one, then keeps
that GPU, including across resume. It never bypasses busy checks. If none is idle,
it exits with code 2 and retains completed CPU results. Rerun the same command
when a GPU becomes available. Do not run other tests or timing jobs on that card.
It checks compute PIDs, utilization and memory before every GPU child and checks
again afterward; external processes, excess resident memory or a failed device
query after a child invalidate that collection. A utilization sample alone can
still include the child's own completed kernels; it does not invalidate results.
Before the next child, the controller allows up to five seconds for this sampled
utilization to clear, but never waits through an observed external process.
Boundary checks cannot prove there was no interference during a child.
The controller and FLA benchmark entry points set `CUDA_VISIBLE_DEVICES` to the
selected physical GPU's UUID, avoiding numeric CUDA ordinal drift. Nested GPU
selection resolves the inherited UUID mask back to the physical index and keeps
the idle checks. Before the first GPU job, the controller also compares CUDA
device 0's UUID with `nvidia-smi`'s physical GPU UUID. An inaccessible device or a
mismatch stops execution. The manifest retains the verified UUID. On this host,
physical GPU 1 currently fails CUDA initialization despite appearing idle in
`nvidia-smi`; physical GPU 4 passed UUID verification for the latest run.

The CPU, GPU and hardware artifact test phases gate later measurements: a failing
test stops the run. MI250 artifact tests skip on H20; the H100-compatible fixture
may run on H20, so the manifest's actual hardware must be included in any report.
Figure 11 compiles forward and backward independently. It records known compiler
resource limits and unsupported feature sizes as missing metrics with explicit
reasons; other compiler/runtime failures still fail the artifact gate. The 1 GiB
workspace limit is retained. Artifact completion therefore does not mean all
Figure 11 shapes or gradients are supported. CSVs, PDF and per-direction
`h100-results/measurements.json` remain under `artifacts-h100/`; the root report
includes these limits and retains `hardware_measurements.json`. Figure 11 timing
has no independent in-run oracle and is marked `MEASURED_UNVALIDATED`.

Use the same output directory and arguments to resume. Collected jobs are
validated and skipped. Failed jobs require `--retry-failed`; prior attempt logs
are kept. A timed-out or interrupted child process group is terminated. The
default per-job timeout is four hours (`--timeout` in seconds); stage workloads
also retain the existing 900-second child timeout. Each retry clears its old
primary result file before execution. Sources, project configuration, FLA
sources, interpreter and arguments must match the original manifest; otherwise
use a new output directory. Commit any implementation fixes before starting a
new measurement run.

`--stages cpu gpu-tests ...` selects stages for a separate run directory. A
benchmark-only selection deliberately bypasses the default test gates and should
only be used with independently retained test evidence. Do not share one output
directory between controllers; an advisory lock rejects concurrent use.

## Results and exit status

The root contains `manifest.json`, `state.json`, `report.md`, `matrix.json`,
`matrix.csv`, and `tests.json`. Each job has an attempt log and its runner's raw
JSON/CSV/plots or pytest JUnit XML. Generalization reports are also linked by job
name in the root report. The matrix preserves failed, rejected, unavailable,
semantically incomparable and numerically invalid cases; only accuracy-qualified
FLA pairs from collected jobs expose speedups. Speedup is FLA/Meta; >1 means Meta
is faster. This script does not infer a PR-base regression or run a second git
revision. That requires a separate baseline worktree and matching workloads.

Exit 0 means every selected job collected its expected artifacts and returned
zero. Exit 1 means failed/incomplete collections. Exit 2 means a test gate, GPU
availability check, configuration check or user interruption stopped the run.
A benchmark child can return zero while recording `INCORRECT`, `REJECTED`,
`NOT_COMPARABLE` or a backend failure; these statuses remain in the report.

Rebuild the report without executing tests or benchmarks:

```bash
.venv/bin/python -m testing.run_pr_matrix report \
  --output-dir benchmark_results/pr_matrix_full
```

Raw result directories are gitignored. Attach/archive them with the PR, and use
the generated report and matrix for the PR's coverage and performance tables.

## Retained results

The single retained full-run snapshot is `benchmark_results/pr_matrix_full/`.
Old numbered runs, partial smoke/plan runs and temporary resume launchers have
been removed. All inherited job directories are real files, so this snapshot
has no dependency on a deleted run. Other standalone benchmark experiments are
separate from this matrix.

The snapshot contains 900 collected jobs, 7,080 benchmark records and 1,278 test
records. Three composition jobs use an 8 GiB workspace target; the other jobs
retain their original 1 GiB target. These overrides are recorded in
`manifest.json` and `consolidation.json`. Collection does not imply numerical
correctness: invalid, rejected and unavailable backend results remain recorded.
See [the results analysis](pr-matrix-results.md) for the findings and limitations.

Historical commands and source fingerprints remain in the snapshot for audit.
The consolidated snapshot is for reporting; its mixed workspace configuration
and historical sources cannot be resumed as a fresh run of the current checkout.
The `report` command above only rebuilds reports and does not execute GPU work.
For future measurements, clear the old snapshot before creating its replacement;
use the same unnumbered output path. `./bench_full.sh --gpu N` launches the full
profile from the project directory; omission of `--gpu` selects an idle GPU.
