# Stateful Operator expression generalization tests

This suite measures how far an accepted Algorithm IR can change while keeping
public-call latency near a simple linear-attention control. It generates synthetic
programs directly from typed DSL nodes, without selecting a named model or
requiring FLA. It complements the four fixed compositions in
`testing/benchmark_stateful_compositions.py`.

## Mutation space

The shared level-zero recurrence has a matrix State, one scalar log decay, one
outer-product write and one query contraction:

```text
S[t] = exp(g[t]) * S[t-1] + k[t] v[t]^T
y[t] = q[t]^T S[t] / sqrt(K)
```

| Family | Meaning of level L |
| --- | --- |
| `expression` | Add L layers to each key, value and query expression. A seeded generator chooses affine, sigmoid and exponential compositions. |
| `propagation` | Add L feature gates, alternating key/value axes with a seeded starting axis. |
| `rank` | Add L ordered rank-one propagations on the key axis, with independently supplied left/right vectors and coefficients. |
| `write` | Add L independent outer-product writes; divide write keys by the total write count to keep State scale bounded. |
| `readout` | Add L independent query contractions of the same State. |
| `combined` | Apply all five mutations at level L. |

Every level preserves the input layout, State dimensions, initial State, output
dtype and common input values. Input generation is seeded per input name, so
adding a tensor does not change the shared control tensors. Default mutation
seeds 1701 and 2028 exercise different expression choices and gate-axis order.
Higher levels represent more authored operations, not a semantic distance metric.
The JSON records actual expression-node occurrences, propagation count, rank-one
count, writes, readouts and input count. Shared expression nodes count at each
use; this is not an estimate of generated instructions or FLOPs.

The suite covers one matrix State and its input-driven affine transitions and
readouts. It does not establish generalization for arbitrary future algorithms,
nonlinear State feedback, multiple coupled States, staged scans, packed layouts
or generated backward. The default GPU feature sizes are 64 by 64; current target
specialization accepts feature sizes 64 and 128. Unsupported sizes are recorded
as rejections.

## Correctness and timing contract

Each control and variant first runs against the independent FP32 CPU token
interpreter shared with the Stateful plan benchmark. It applies authored
propagations and writes in order and checks every output and the final State.
The initial State is nonzero. Readouts are BF16; gate inputs and State are FP32.
Outputs use `rtol=0.03, atol=0.001`; final State uses
`rtol=0.02, atol=0.001`. These tolerances are serialized alongside maximum
absolute errors. Incorrect results are recorded as `INCORRECT` and are not timed.
Compilation/resource failures are `REJECTED`; other runtime failures are `ERROR`.

Measurements cover **one public forward call returning all outputs and final
State**. Compilation, first-call code generation, oracle evaluation and warmup
are excluded. Internal layout normalization, preparation and output allocation
remain inside the call. Each of ten default measurement batches contains five
calls per implementation. Control and variant alternate order between batches.
CUDA events and synchronized wall time are reported separately. Both retain raw
samples, per-batch medians, and the median of paired variant/control ratios.
Peak incremental allocated bytes are measured on a warmed call with live inputs
and returned outputs; they are not reserved memory or total device memory.

Default stability requires both:

- Median paired variant/control latency ratio at most **1.25**.
- Paired ratio interquartile range divided by its median at most **0.10**.

Use `--max-slowdown`, `--max-jitter` and `--metric cuda_ms|wall_ms` to change the
criterion. JSON distinguishes the latency budget from repeatability. Both seeds
must pass at each level for the reported envelope to advance. An incorrect,
rejected, noisy or slow point closes the contiguous sequence of tested levels;
a later recovery does not erase an earlier failure. A passing largest level is
only a lower bound on the stable range. Untested intermediate levels are not
inferred to pass.

Increasing rank, write count or output count adds real work. Ratios therefore
measure preservation of end-to-end latency, not equal-FLOP compiler efficiency.
A large ratio can come from additional work, expression preparation or a change
of lowering. Inspect the recorded runtime family, representation, rewrites,
workspace estimate and memory alongside the ratio.

## Run

From the repository root, inspect the profile matrix without a GPU:

```bash
.venv/bin/python testing/benchmark_stateful_generalization.py --dry-run \
  --levels 0 1 2 4 8 --seeds 1701 2028 --lengths 65 512 2048
```

Run the full default matrix on an idle H20:

```bash
.venv/bin/python testing/benchmark_stateful_generalization.py --gpu 3 \
  --levels 0 1 2 4 8 --seeds 1701 2028 --lengths 65 512 2048 \
  --output-dir benchmark_results/generalization
```

For a short smoke measurement:

```bash
.venv/bin/python testing/benchmark_stateful_generalization.py --gpu 3 \
  --levels 0 1 --seeds 1701 --lengths 65 \
  --warmup 2 --repetitions 3 --batches 4 \
  --output-dir benchmark_results/generalization-smoke
```

The runner writes `records.json`, `summary.csv` and `report.md` after every
measurement. JSON includes source hashes, commit/worktree status, GPU inventory,
software versions, arguments and raw samples. `report.md` presents the tested
stable envelope; CSV contains one row per point. A recorded failure does not stop
the rest of the matrix. The process finishing successfully means the matrix was
collected, not that every program was correct or stable.

Correctness and benchmark-contract tests:

```bash
.venv/bin/python -m pytest tests/unit/test_stateful_generalization.py -q
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest \
  tests/functional/test_stateful_generalization.py --run-gpu -q
```

GPU correctness tests exercise sequence length 65, multiple batches, nonzero
initial State, and a two-call continuation split at token 33. Returned State is
made contiguous before reuse, as required by the public input contract.
Performance thresholds are evaluated by the benchmark report, rather than
asserted against shared-runner timing in ordinary unit/functional tests.

## Initial H20 measurement

The checked-in [records](measurements/stateful-generalization-h20/records.json),
[CSV](measurements/stateful-generalization-h20/summary.csv),
[envelope report](measurements/stateful-generalization-h20/report.md) and
[manifest](measurements/stateful-generalization-h20/manifest.json) contain a
96-point run: six families, levels 0/1/2/4, seeds 1701/2028, and T=512/2048.
B=1, H=4, K=V=64; workspace limit 1 GiB; ten paired batches with three warmups
and five repetitions. Physical GPU 3 was selected with `CUDA_VISIBLE_DEVICES=3`
and appears as logical GPU 0 in the runner. The initial inventory recorded 2%
utilization on that device; other devices were shared. PyTorch was
2.11.0+cu128, CUDA 12.8 and TileLang 0.1.15.

There were **94 correctness passes and two incorrect points**: value-axis
propagation at level 1, seed 1701, at both sequence lengths. All six families
exceeded the default 1.25x latency budget at level 1, so every measured envelope
contains only level 0. All level-zero self-controls passed the stability check.

At T=2048, the following ranges cover the two seeds, except the propagation
level-1 entry, which has only one correct seed. Ratios above one mean slower
than the same-shape level-zero control.

| Mutation | Level 1 / control | Level 4 / control | Level-1 lowerer |
| --- | ---: | ---: | --- |
| Input expressions | 4.01–4.09x | 6.69–6.77x | `tilelang-elementwise` |
| Feature propagation | 3.39x; other seed incorrect | 4.19–4.21x | `tilelang-elementwise` |
| Ordered rank-one propagation | 11.13–11.36x | 15.91–15.94x | `tilelang-factored-recurrence` |
| Additional writes | 11.00–11.04x | 18.16–18.40x | `tilelang-elementwise` |
| Additional readouts | 9.21–9.28x | 11.62–11.84x | `tilelang-elementwise` |
| Combined | 15.85–17.14x | 59.47–61.47x | `tilelang-factored-recurrence` |

The control selected `tilelang-scalar-factorized`. For the input-expression
family, level 1 changes the authored expression count from 9 to 25–27 without
adding inputs, State dimensions, writes or outputs, and changes the selected
lowerer. That is a useful first performance cliff to investigate. The other
families also add mathematical work; their ratios alone do not isolate compiler
overhead. These measurements establish a narrow tested latency envelope, even
though most generated programs remain numerically correct. They do not justify
a claim of stable performance for arbitrary future linear-attention algorithms.

Reproduce this particular matrix:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python testing/benchmark_stateful_generalization.py \
  --gpu 0 --levels 0 1 2 4 --seeds 1701 2028 --lengths 512 2048 \
  --output-dir benchmark_results/generalization_h20
```

## Known correctness boundary

At `family=propagation, level=1, seed=1701`, scalar decay plus a value-axis feature
gate produces a final State outside the independent token-oracle tolerance on
H20. A short B=1/H=2/T=65/K=V=64 probe observed a maximum absolute State error
of approximately 0.014, against an absolute tolerance of 0.001. The generated
CPU recurrence passes. The GPU regression is a strict expected failure for this
specific profile, allowing an eventual correction to surface as XPASS. The
benchmark retains the point as `INCORRECT`, including selected lowerer and
correctness error; it never reports that point as performance-stable. This test
addition does not change the compiler implementation.
