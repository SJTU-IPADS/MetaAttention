# Prove H20 Correctness and Performance

Labels: `ready-for-agent`

## Goal
Produce reproducible H20 correctness, composition, packed-sequence, and performance evidence for the proposed compiler.

## Scope
- Verify product-family plus CUDA-capability identity and rejection of unsupported targets/MIG identity.
- Test four real profiles and one model-independent pressure profile through `StatefulOperator → ExecutionResult`.
- Pressure profile: ordered `AxisScale → RankOnePropagation(left != right) → AxisScale`, multiple injections/readouts, `(64,128)` State, grouped heads, packed variable lengths, empty sequences, inactive tails, and compiler-derived backward.
- Benchmark old/new real-profile baselines on an exclusive H20 using interleaved paired batch medians and 95% bootstrap confidence intervals.

## Non-goals
No H100 extrapolation, CPU production claim, first-compile latency gate, or pressure-profile legacy baseline.

## Dependencies
Issues 01–07.

## Acceptance
Forward, backward, packed, continuation, alias, dtype, layout, and error suites pass on H20. Latency and peak-memory ratio CI upper bounds are at most `1.10`; an interval wholly above `1.10` blocks, and crossing intervals receive more batches. Compile time is recorded. The pressure profile meets predeclared absolute timeout/memory limits. Full existing functional GPU tests pass.
