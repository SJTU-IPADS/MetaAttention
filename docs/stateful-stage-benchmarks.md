# Public scan-stage benchmarks against FLA

`testing/benchmark_fla_stateful.py` now includes these three **attention
operators** in its default family registry, alongside the existing 22 families.
`testing/benchmark_fla_stages.py` remains a focused diagnostic for nonzero
supplied caches, explicit input dtype/dimensions and separately measured VJPs.
These entries describe attention operators, not model layers:

| CLI family | Public Meta factory | FLA prefill | FLA recurrent |
| --- | --- | --- | --- |
| `preconditioned-kda` | `examples.preconditioned_delta.preconditioned_delta(kind="kda")` | `chunk_precond_kda` | `fused_recurrent_precond_kda` |
| `preconditioned-gdn` | same, `kind="gdn"` | `chunk_precond_gated_delta_rule` | `fused_recurrent_precond_gated_delta_rule` |
| `lightnet` | `examples.lightnet_stable.lightnet_stable` | `chunk_lightnet` | `fused_recurrent_lightnet` |

The runner invokes the **public compiled stage callable**, not private kernels.
It records the selected lowerer and runtime family for each child, separately
for forward and backward. LogAddExp is a dedicated child runtime and has no
ordinary candidate lowerer; its `selected_lowerer` is null, not a missing
benchmark. A selected lowerer describes compilation, not proof that the
same native executor runs during autograd.

## Unified benchmark interface

Use the main runner for the same matrix, timing and result conventions as the
other families:

```sh
.venv/bin/python testing/benchmark_fla_stateful.py \
  --families preconditioned-kda,preconditioned-gdn,lightnet \
  --batches 1 2 --heads 1 8 --seqlens 65 2048 8192 \
  --contract all --fla-baseline both --timing both \
  --workspace-gib 8 --warmup 5 --repetitions 30 \
  --fla-source ../flash-linear-attention --gpu 0 \
  --output-dir benchmark_results/staged-unified
```

Omitting `--families` selects all 25 registered families. The unified runner
uses BF16 inputs and the default initial State of each authored operator:
zero ATK/memory for preconditioned attention, and `z=-inf` with zero memory
for LightNet. `output-only` requests no finals; `final-state` validates every
State; `continuation` times prefix and suffix public calls, passes both actual
prefix caches to the suffix, and includes output concatenation. Noncontiguous
Meta final views are made contiguous at this boundary, inside the timed call.
Continuation requires T>=2; use a separate output-only/final-state run for T1.

All invocations, profiling and Graph capture run in `torch.inference_mode()`.
External FLA layout conversion, gate construction, normalization and state
layout conversion are included in the public interval, matching the other
families in this runner. Compilation and independent CPU oracle execution are
excluded. Public CUDA-event, synchronized CPU-wall, Graph replay, CUPTI and
allocator measurements use the existing shared helpers. The output directory
contains `manifest.json`, `records.json`, `summary.csv` and comparison plots.
Schema 6 remains unchanged; staged records additionally include
`selected_child_lowerers` and `staged_workspace_estimate_bytes`, rather than
inventing a single child plan or treating a workspace bound as measured memory.

FLA schedule selection is explicit and missing APIs remain `UNAVAILABLE`.
Public chunk KDA's q/k normalization differs from the authored operator, so
its prefill records are `NOT_COMPARABLE`, with correctness errors and the
semantic reason retained. They produce no speedup or plotted winner. GDN and
LightNet expose matching two-State final/continuation contracts. The existing
CSV/plot eligibility checks apply to every family.

The unified interface is a forward benchmark. The focused runner below retains
separate VJP measurements and nonzero supplied initial caches; its timings use
prebound external layouts and therefore must not be mixed with unified public
measurements.

## Focused stage diagnostic

From the repository root, with the CUDA/TileLang environment installed:

```sh
.venv/bin/python testing/benchmark_fla_stages.py \
  --gpu 1 --family preconditioned-kda,preconditioned-gdn,lightnet \
  --seqlens 1 64 2048 8192 --heads 1 --dim 64 --dtype bfloat16 \
  --workspace 8589934592 --warmup 3 --reps 20 \
  --fla-path ../flash-linear-attention \
  --output /tmp/fla-stages.json
```

`--families` is an alias for `--family`. `--workspace` is **bytes**, not MiB.
`--heads` and `--seqlens` accept lists. Batch is deliberately fixed to one;
K/V dimensions and K/V head counts are equal. T1 is labeled decode with
nonzero supplied caches; larger T is labeled prefill with supplied caches.
This is a single invocation workload, not a preceding prefill followed by
an entire decode loop. `--no-backward` requests forward-only measurements.
Supported input dtypes are FP32, BF16 and FP16; unsupported FLA kernel dtypes
are recorded as failures, not coerced into another dtype.

The physical GPU is checked for idle utilization, memory and compute PIDs
**before importing torch**. An explicit busy `--gpu` is rejected; the runner
never silently uses another card. It respects an existing numeric
`CUDA_VISIBLE_DEVICES` mask. Run long jobs in the background with stdout and
stderr redirected to a log; each backend announces its start. The JSON file
is rewritten after every completed workload. Each workload runs in a fresh
child process (`--case-timeout`, default 900 seconds); completed backend
checkpoints survive a child timeout or fatal kernel/OS exit. Unfinished
backends are marked `CHILD_TIMEOUT` or `CHILD_FAILED`, not omitted. The
controller checks the chosen GPU again before each child; a newly occupied
GPU produces `GPU_BUSY_REJECTED` records rather than switching GPUs. GPU
snapshots at both boundaries flag observed external contention; absence of
a boundary PID is not a guarantee that no interference occurred in between.

## Numerical contract

All three implementations receive the same quantized q/k/v inputs, FP32
gates and FP32 nonzero initial states. In the inspected FLA checkout, public chunk
KDA unconditionally L2-normalizes q/k while recurrent KDA and the authored
operator use original q/k. This baseline semantic difference is explicitly
recorded, not corrected by changing inputs or replacing the public API. Preconditioned ATK state is positive;
center is fixed to -0.2, x to 1.5, epsilon to 1e-6, and query scale to
K**-0.5. GDN normalizes q/k; KDA does **not**. LightNet consumes raw logits,
not centered keys, and its initial log-normalizer is finite. Layer projections,
SiLU, convolutions, output norm and gating are outside both intervals.

Primary validation is an independent FP32 **CPU token recurrence**, not
agreement with a FLA optimized kernel. Every available forward backend is
checked against output **and both final states**. The JSON reports max absolute
and RMS error per tensor, with `rtol=0.02, atol=0.005`. Available FLA outputs
also get an explicit Meta pair-difference record. A numerical disagreement
is `NUMERICAL_MISMATCH`, and its measured latency is retained as diagnostic
evidence, **not** an accuracy-qualified speedup. No speedup/winner is emitted.
This tolerance is intended for mixed precision exploratory workloads, not a
proof of equivalence across arbitrary adversarial inputs.

Backward uses unit cotangents on output and both final states. An independent
full VJP oracle checks T<=64 with `rtol=0.04, atol=0.02`. Longer backward
measurements explicitly report `gradient_oracle.status=NOT_CHECKED` and
`status=MEASURED_UNVALIDATED` unless their forward contract already mismatches;
forward correctness is not represented as gradient correctness. Meta exposes
center and per-token scale derivatives that the FLA APIs do not expose;
these constants are excluded from **both** timed VJPs for an equal gradient
input set. Each backend records `excluded_gradients` and its actual gradient
inputs. The CPU factory tests additionally verify Meta's full exposed VJP. Remaining input and initial-state gradients are
compared against the oracle in canonical head-first layout. FLA recurrent
backward is never substituted for chunk backward.

## Timing and memory contract

- External tensors are prebound: contiguous **B,H,T,D** for Meta and
  **B,T,H,D** for FLA. External transpose/materialization is excluded on
  **both** sides. FLA backward leaves are already in native layout, so the
  VJP interval does not backpropagate through an external layout-copy graph.
- Internal preprocessing **is included**: ATK/LightNet gate construction,
  normalization, public input/state validation, dispatch, stage bindings,
  intermediate contiguous copies, allocations and kernel launches.
- Forward is compiled with backward disabled and invoked entirely in
  `torch.inference_mode()` to avoid grad-enabled differentiable execution.
  The CPU-test fallback environment is explicitly rejected for GPU runs.
- Compilation/JIT and correctness are outside timing. Backward forward-graph
  construction and cotangent allocation are also outside timing. Backward
  is **VJP-only**, repeatedly applying `autograd.grad(..., retain_graph=True)`
  to the same graph; it is not full training-step or forward+backward latency.
- CUDA-event and synchronized CPU-wall medians surround the same warmed
  invocation. There is at least one untimed priming call even with warmup=0
  to exclude lazy backward JIT. Both raw sample arrays are retained. No CUDA Graph capture,
  CUPTI kernel-only timing or subtraction of Python/public overhead is used.
- A separate warmed call precedes timing for allocator measurement. JSON
  includes resident allocated bytes, absolute peak allocated bytes and peak
  incremental allocated bytes. Backward graph storage is resident. CUDA
  context/Triton/driver allocations outside PyTorch are not included.
- The staged workspace estimate is the public compiled `workspace_bytes`
  bound, not measured peak memory. Forward and backward estimates are
  recorded separately. FLA has no equivalent public estimator here;
  its allocator peak must not be called a staged workspace estimate.

Each family/T/H record independently attempts Meta forward, FLA chunk forward,
FLA recurrent forward, Meta backward and FLA chunk backward. Compilation,
workspace/resource rejection, unavailable APIs and kernel exceptions appear
with status, exception type, message and available structured details. One
failed backend does not prevent attempts of other backends or later
workloads. There is no fallback to another family, dtype, schedule, CPU or
private kernel. A CUDA context fault may make later backends in the same child fail; those
failures remain visible. A fresh child prevents that context from poisoning
subsequent workloads.
