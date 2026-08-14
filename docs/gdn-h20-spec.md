# H20 Gated Delta Rule Adaptation

## Problem Statement

MetaAttention currently supports scalar-decay recurrent linear attention through a shared LinearAttentionEngine, but it cannot express QLA's Gated Delta Rule. GDN changes the recurrent state transition by adding beta-weighted delta correction, a causal KKT solve, and explicit initial and final state. Users targeting H20 therefore cannot define, lower, train, or benchmark QLA-compatible GDN through MetaAttention without introducing a dedicated path.

The upstream FlashQLA kernels do not solve this directly: their optimized implementations target SM90 and newer architectures, while H20 is SM89. Reusing the Hopper warp-specialized kernels would rely on unavailable TMA and warpgroup behavior. MetaAttention needs an H20 GDN Adapter that preserves FlashQLA's mathematical and training behavior without pretending that the Hopper schedule is portable.

## Solution

Add a dedicated GDN Engine for H20/sm89. The module will expose one high-level training interface, lower fixed-length head-first GDN inputs into staged TileLang kernels, and return the GDN output plus an optional final state. It will remain separate from LinearAttentionEngine because GDN has a different state recurrence rather than a different elementwise modifier configuration.

The first release will implement gate cumsum, causal KKT solve, state/output evaluation, and backward as separate H20-compatible kernels. It will use FlashQLA's MIT-licensed mathematical reference and observable test behavior as the compatibility definition, with attribution, but will not reuse Hopper-only warp-specialized implementation details.

## User Stories

1. As an H20 model developer, I want to construct a dedicated GDN Engine, so that I can run QLA-compatible Gated Delta Rule without changing the existing scalar-decay LinearAttentionEngine.
2. As an H20 model developer, I want query and key inputs in MetaAttention's head-first `[B,Hk,T,128]` layout, so that I do not need token-first transposes or copies.
3. As an H20 model developer, I want value inputs in `[B,Hv,T,128]` layout, so that each value head has an independent recurrent state.
4. As an H20 model developer, I want log-space gate and beta inputs in `[B,Hv,T]` layout, so that the GDN recurrence receives its complete control inputs.
5. As an H20 model developer, I want `Hv` to be an integer multiple of `Hk`, so that each query/key head maps deterministically to one or more value/state heads.
6. As an H20 model developer, I want grouped value attention to reduce query and key gradients back across their mapped value heads, so that training gradients match the defined GVA semantics.
7. As an H20 model developer, I want a missing initial state to mean an all-zero state, so that ordinary full-sequence training does not require state allocation by the caller.
8. As an H20 model developer, I want to provide an FP32 initial state in `[B,Hv,128,128]` layout, so that I can continue a recurrent computation from an earlier segment.
9. As an H20 model developer, I want to request an FP32 final state, so that I can continue the recurrence in a later segment.
10. As an H20 model developer, I want final state production to be optional, so that calls which only need output avoid unnecessary state materialization and return values.
11. As a training user, I want loss gradients from both output and final state to participate in backward, so that segmented recurrent training remains mathematically complete.
12. As a training user, I want gradients for query, key, value, gate, beta, and a provided initial state, so that all trainable GDN inputs can be optimized.
13. As a training user, I want query, key, value, and output to use BF16, so that the operator matches the first-release H20 precision and throughput contract.
14. As a training user, I want gate, beta, state, and state accumulation to use FP32, so that gate accumulation, KKT computation, and recurrent state updates remain numerically stable.
15. As a model developer, I want an optional scalar query scale, so that model-specific scaling can be represented without an extra tensor input.
16. As a model developer, I want the default scale to be `128**-0.5`, so that omission follows the QLA reference behavior.
17. As a model developer, I want the engine to reject sequence lengths not divisible by 64, so that the first release never silently truncates or implicitly pads recurrent input.
18. As a model developer, I want invalid head counts, dimensions, layouts, dtypes, state shapes, and devices rejected before kernel execution, so that failures identify contract violations instead of producing incorrect output.
19. As an H20 user, I want the engine to reject non-sm89 hardware explicitly, so that unsupported devices never compile or cache a misleading adapter.
20. As a maintainer, I want GDN lowering isolated from scalar-decay linear-attention lowering, so that adding QLA does not change existing retention or Mamba2 behavior.
21. As a maintainer, I want gate cumsum, KKT solve, state/output, and backward staged separately, so that numerical and gradient defects can be localized before performance fusion is attempted.
22. As a maintainer, I want forward to save gate cumsum and the KKT result while backward rebuilds chunk states, so that training avoids both repeated KKT solve and excessive saved-state memory.
23. As a maintainer, I want Q/K normalization left outside the GDN Engine, so that normalization policy and its gradients remain owned by the model layer.
24. As a maintainer, I want the H20 adapter checked against an in-repository FP64 reference, so that correctness does not depend on installing or executing FlashQLA.
25. As a maintainer, I want output, final state, and every supported gradient checked with a relative L2 error limit of 2%, so that BF16 reduction differences do not hide substantial algorithm errors.
26. As a maintainer, I want one-chunk and multi-chunk scenarios covered, so that both local causal computation and recurrent state passing are verified.
27. As a maintainer, I want zero and nonzero initial-state scenarios covered, so that state initialization and propagation are independently verified.
28. As a maintainer, I want equal-head and grouped-value-head scenarios covered, so that both direct and reduced GVA gradient paths are verified.
29. As a maintainer, I want beta values near zero and one covered, so that weak and strong delta updates are verified.
30. As a maintainer, I want gates near zero and substantially negative gates covered, so that low-decay and high-decay behavior is verified.
31. As a maintainer, I want state-gradient-only and output-gradient-only losses covered, so that both backward entry gradients are observed independently.
32. As a performance engineer, I want H20 kernel latency, end-to-end latency, peak memory, and first compilation time recorded on representative shapes, so that later fusion work has a reproducible baseline.
33. As a performance engineer, I want first-release performance measurements treated as baselines rather than pass/fail comparisons with Hopper FlashQLA, so that architecture differences do not create an invalid acceptance criterion.
34. As a reviewer, I want the adaptation delivered in independently verifiable milestones, so that reference behavior, forward state semantics, backward gradients, and performance documentation can be reviewed separately.
35. As a future maintainer, I want FlashQLA's MIT attribution retained wherever its formulas or test behavior directly informed the implementation, so that provenance remains explicit.

## Implementation Decisions

- Build a dedicated GDN Engine. Do not add an algorithm mode or GDN-specific optional parameters to LinearAttentionEngine.
- Use one public behavioral seam: invoking the GDN Engine with query, key, value, gate, beta, optional scale, optional initial state, and an output-final-state flag. Functional tests cross this seam rather than calling individual generated kernels.
- The public tensor contract is head-first:
  - query and key: `[B,Hk,T,128]` BF16;
  - value: `[B,Hv,T,128]` BF16;
  - log-space gate and beta: `[B,Hv,T]` FP32;
  - output: `[B,Hv,T,128]` BF16;
  - optional initial and final state: `[B,Hv,128,128]` FP32.
- Require `Hv % Hk == 0`. Each query/key head serves `Hv/Hk` value and state heads. Backward sums query/key gradients across those mappings.
- Require `T > 0` and `T % 64 == 0`. Reject unaligned lengths explicitly. Do not truncate or implicitly pad.
- Support only CUDA H20/sm89 in the first release. Reject CPU, ROCm, H100, and other CUDA capabilities at GDN Engine creation before code generation.
- Fix head key/value dimensions at 128 and chunk size at 64 for the first release.
- Accept an optional Python float scale. Default it to `128**-0.5` and specialize it as a compile-time kernel constant.
- A missing initial state means zero state. Produce final state only when requested.
- Final state participates in autograd. Backward consumes both output gradient and final-state gradient and returns an initial-state gradient when initial state was provided.
- Do not implement Q/K L2 normalization inside the GDN Engine. Callers must supply already prepared query and key tensors.
- Implement an H20 GDN Adapter using staged TileLang kernels:
  1. chunk-local gate cumsum;
  2. beta-weighted causal KKT construction and lower-triangular solve;
  3. recurrent state preparation and output computation;
  4. backward state reconstruction and gradients;
  5. reverse gate cumsum and GVA query/key gradient reduction.
- Use FlashQLA's mathematical reference and observable tests as the compatibility source. Do not port Hopper-only TMA, warpgroup specialization, or barrier schedules to sm89.
- Retain applicable MIT copyright and attribution in adapted implementation and reference material.
- Save gate cumsum and the KKT result from forward. Backward rebuilds chunk states and does not save `w`, `u`, corrected values, or all chunk states from forward.
- Reuse existing MetaAttention infrastructure where semantics match: generated-module hashing and loading, meta-tensor validation conventions, TileLang compilation, GEMM patterns, and test markers.
- Do not add generic matrix, triangular-solve, or state-scan nodes to the symbolic modifier graph in the first release. There is one concrete adapter; a generic seam would expose more interface than leverage.
- Do not route GDN through score modification, online softmax, or the scalar-decay linear template. Those paths cannot represent beta/KKT delta correction.
- Deliver in four independently reviewable milestones:
  1. FP64 reference and public-contract tests;
  2. H20 forward, initial state, and final state;
  3. backward for all supported gradients;
  4. benchmark baselines, public documentation, attribution, and cleanup.

## Testing Decisions

- Test at the GDN Engine interface, the highest behavioral seam. The kernel-stage functions are implementation details and are not separately asserted except where a focused compilation smoke test is needed to diagnose an adapter failure.
- The reference oracle is an in-repository FP64 PyTorch implementation derived from the published FlashQLA GDN reference behavior. Tests do not import FlashQLA as a runtime dependency.
- Forward tests compare output and requested final state against the FP64 reference using relative L2 error no greater than 0.02.
- Backward tests compare `dq`, `dk`, `dv`, `dg`, `dbeta`, and `dinitial_state` against the FP64 reference using relative L2 error no greater than 0.02.
- Test one chunk and multiple chunks to cover local causal behavior and recurrent state passing.
- Test zero initial state and caller-provided nonzero state.
- Test final-state-disabled calls, final-state-enabled calls, output-only loss, final-state-only loss, and combined loss.
- Test `Hk == Hv` and at least one `Hv > Hk` GVA case.
- Test representative gate regimes and beta values near their useful boundaries without relying on nondeterministic random extremes.
- Test custom scale and default scale separately.
- Contract tests assert explicit failures for:
  - non-sm89 device;
  - non-BF16 query/key/value;
  - non-FP32 gate/beta/state;
  - `K` or `V` other than 128;
  - mismatched batch or sequence dimensions;
  - `Hv % Hk != 0`;
  - invalid state shape;
  - zero or non-64-aligned sequence length;
  - noncontiguous or unsupported-stride tensors if the adapter requires contiguous storage.
- CPU-safe tests cover reference behavior, public input validation that does not require device inspection, generated-code contracts, and importability.
- H20-marked functional tests compile and run the actual generated kernels. A test skip on absent H20 is acceptable locally but is not evidence of kernel correctness.
- Prior art is the repository's functional reference/parity suite for existing attention engines and its GPU fixture/marker structure. GDN uses a tighter relative-L2 standard modeled on the upstream FlashQLA unit tests rather than the legacy linear-attention `0.1` tolerance.
- Performance verification records, but does not gate on, representative H20 kernel latency, end-to-end latency, peak allocated memory, and first compilation time. Measurement configuration, warmup, repetitions, dtype, and shapes must be included with the results.
- Each permanent behavior change must be exercised through the actual GDN Engine callable; generated source text alone is not sufficient proof.

## Out of Scope

- Existing scalar-decay gated retention, RetNet, Mamba2, and LinearAttentionEngine behavior changes.
- Variable-length packed sequences and `cu_seqlens`.
- Automatic intra-card context parallelism, warmup chunk selection, state correction, or CP caches.
- Sequence lengths not divisible by 64.
- Internal or automatic padding.
- FP16, FP64 kernel inputs, or mixed Q/K/V input dtypes.
- Key or value head dimensions other than 128.
- Chunk sizes other than 64.
- CUDA architectures other than H20/sm89, including SM90 Hopper and newer Blackwell variants.
- ROCm and MI250 support.
- FlashQLA's Hopper warp-specialized schedule or performance parity.
- Q/K L2 normalization inside the GDN Engine.
- Decode-specific state-cache interfaces beyond explicit initial/final state passing.
- A general-purpose scan, matrix, or triangular-solve symbolic IR.
- Runtime selection between naive and fused GDN implementations.
- Performance fusion beyond the staged first-release kernels.

## Further Notes

- QLA means the upstream FlashQLA implementation of Gated Delta Rule in this project. It must not be used as a synonym for the existing simple GLA/gated-retention path.
- H20 is SM89. The upstream FlashQLA architecture dispatcher does not support SM89, so successful installation of FlashQLA would not make its kernels valid for this target.
- The existing scalar-decay linear template computes its chunk count with integer division and does not explicitly reject a tail chunk. The GDN Engine deliberately avoids that silent behavior by making 64-token alignment a checked public contract.
- The H20 adapter is correctness-first. Profiling data from the staged implementation will determine whether later fusion is justified and where the seam should move.
- Domain language is defined in the repository context glossary. The architectural rationale for a dedicated H20 GDN Engine is recorded in the corresponding ADR.
