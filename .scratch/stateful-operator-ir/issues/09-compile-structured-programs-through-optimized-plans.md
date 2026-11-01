# Compile Structured Stateful Programs Through Optimized Mathematical Plans

Labels: `ready-for-agent`

Status: `proposed`

## Problem Statement

Stateful Operator needs competitive H20 performance without becoming a catalog of model-specific or whole-program kernels. A legal Scanable Program may contain a previously unseen composition of existing Algorithm IR nodes. Requiring a dedicated operator for every such composition would make compatibility depend on recognizing model names or exact graphs, duplicate semantic implementations, and prevent the generic compiler from serving as a reliable baseline.

The current compiler analysis recognizes a small amount of structure and the current TileLang path selects between coarse implementation forms, but it does not expose a complete boundary between mandatory algebraic simplification, alternative mathematical representations, backend lowering, candidate performance comparison, kernel-parameter autotuning, and compiler-derived backward generation. Compilation also currently depends on concrete example Tensors, which unnecessarily fixes Batch and Sequence metadata that should remain dynamic at invocation time.

The compiler needs to accept every Target-supported Scanable composition in the existing Algorithm IR, simplify it safely, enumerate reusable model-independent Mathematical Plans, lower those plans through independently extensible kernel implementations, and select the fastest valid implementation. Failure to recognize or compile an optimization must not make a legal Scanable Program unsupported: generic dense affine forward and backward lowering remain the semantic baseline.

This specification records the intended design for later development on an H20 machine. It does not claim that the described performance behavior has already been implemented or measured.

## Solution

Compile each validated Algorithm IR program through one compositional pipeline. Canonicalization first removes representation-only differences while preserving Feature-role order, authored noncommutative Propagation order, and floating-point association. Mandatory safe simplification then produces one Simplified IR. A planner enumerates equivalent Mathematical Plans from that Simplified IR; plans describe algorithm structure only and contain neither backend schedules nor kernel parameters.

Registered kernel lowerers convert applicable Mathematical Plans into Lowered Candidates. The compiler always includes a generic dense affine candidate. With optimization enabled, it statically filters and ranks all valid candidates, retains the best three plus the generic baseline, and—when autotuning is enabled—benchmarks them on a versioned compiler-owned workload policy. It selects the candidate with the lowest normalized geometric-mean runtime latency, then autotunes only that candidate's concrete TileLang configuration. Memory and other resources are hard feasibility constraints; among feasible candidates, runtime latency is the primary selection objective.

Compilation becomes metadata-only. Callers provide fixed Head and Feature role sizes, Target, dense or packed mode, and the independent `optimized`, `autotune`, and `backward` choices. Runtime Batch size, dense Sequence length, packed logical-sequence count, packed total-token count, and packed offsets remain dynamic. Invocation supplies actual Tensors and receives the existing immutable ExecutionResult.

When backward generation is requested, the compiler derives canonical Backward IR from compiler-owned VJPs, independently plans and lowers forward and backward, and selects a context-compatible pair. A compiler-owned named context contract describes values produced by forward, required by backward, or explicitly recomputed. The returned callable uses PyTorch autograd only to save context and invoke the generated backward kernel; PyTorch does not derive or replace the compiler's backward semantics.

Successful plan measurements, kernel tuning results, and binary artifacts are cached under identities that separate semantics, Mathematical Plans, lowerers, kernel configurations, context contracts, Target/toolchain state, and workload-policy versions. Failed candidate attempts are not persisted. Specialized failures are summarized in one warning and compilation continues; failure of the generic baseline is a structured compilation error.

## User Stories

1. As a Stateful Operator author, I want previously unseen compositions of existing Algorithm IR nodes to compile, so that compatibility does not require a new named operator.
2. As a Stateful Operator author, I want compatibility defined at the Algorithm IR boundary, so that arbitrary frontend or backend IR recognition is not silently promised.
3. As a model author, I want GLA, RetNet recurrent evolution, selective State evolution, GDN, and new compositions to use the same compiler language, so that model names never become dispatch keys.
4. As a compiler user, I want every Target-supported Scanable Program to have a generic dense affine implementation, so that optimization recognition is not a legality requirement.
5. As a compiler user, I want an optimization miss to select the generic baseline, so that valid programs do not fail merely because they are unfamiliar.
6. As a compiler user, I want Recurrent classification to remain independent of Target capability and optimization availability, so that mathematical and hardware failures stay distinguishable.
7. As a compiler maintainer, I want validation to precede optimization, so that lowerers never receive ill-typed or semantically invalid programs.
8. As a compiler maintainer, I want canonicalization to remove commutative operand-order differences, so that equivalent authored forms share analysis and cache identity.
9. As a compiler maintainer, I want nested Propagation structure flattened and identity Propagations removed, so that representation-only syntax does not obstruct matching.
10. As a compiler maintainer, I want empty Propagation and explicit identity to canonicalize equally, so that a pure accumulator has one canonical meaning.
11. As a compiler maintainer, I want safe zero and one simplifications applied uniformly, so that every lowerer consumes the same reduced structure.
12. As a numerical-kernel author, I want floating-point association preserved, so that optimization does not silently reassociate Algorithm IR expressions.
13. As an operator author, I want authored noncommutative Propagation order preserved, so that optimization cannot reorder State evolution.
14. As an operator author, I want ordered State Feature roles preserved, so that left and right State actions remain unambiguous.
15. As a compiler maintainer, I want mandatory simplification separate from optional representation rewrites, so that correctness-preserving cleanup does not depend on performance decisions.
16. As a compiler maintainer, I want every optional representation rewrite to declare its preconditions and conflicts, so that only legal combinations become Mathematical Plans.
17. As a correctness reviewer, I want every plan rewrite to preserve dtype rules, floating-point ordering guarantees, and compiler VJP semantics, so that a faster plan remains the same program.
18. As a performance engineer, I want adjacent scale fusion represented as a reusable structural optimization, so that every matching program can benefit.
19. As a performance engineer, I want Product Injection/GEMM recognition represented independently of model identity, so that unfamiliar programs can use efficient matrix operations.
20. As a performance engineer, I want Rank-One sequences eligible for factorized representations, so that WY/KKT-style implementations can compete where mathematically valid.
21. As a performance engineer, I want shared State Contractions eligible for readout fusion, so that redundant State traffic can be removed.
22. As a compiler maintainer, I want the dense affine representation always enumerated, so that specialized representations have a stable semantic and performance baseline.
23. As a compiler maintainer, I want Mathematical Plans to contain only algorithm structure, so that hardware schedules and tuning results do not contaminate mathematical identity.
24. As a compiler maintainer, I want applied rewrite identities recorded in each Mathematical Plan, so that plan selection is deterministic and explainable.
25. As a compiler maintainer, I want fused mathematical regions represented by semantic boundaries, so that plans do not hard-code final kernel-launch counts.
26. As a kernel author, I want to add a new implementation of an existing Mathematical Plan without changing the central planner, so that backend growth does not create a monolithic dispatcher.
27. As a kernel author, I want one Mathematical Plan to admit multiple lowerers, so that persistent, hierarchical, and multi-pass implementations can compete.
28. As a compiler maintainer, I want lowerers prohibited from introducing undeclared mathematical rewrites, so that plan and artifact identities remain truthful.
29. As a compiler maintainer, I want new mathematical representations added through planner rules rather than hidden inside kernels, so that equivalence remains reviewable.
30. As a performance engineer, I want Lowered Candidates statically filtered for capability and resources, so that impossible kernels are not benchmarked.
31. As a performance engineer, I want the statically strongest three candidates retained, so that search remains bounded.
32. As a performance engineer, I want the generic baseline added to the measured set even when it is not statically top-ranked, so that specialization must demonstrate actual benefit.
33. As a performance engineer, I want each measured candidate to receive a small fixed parameter-probe budget, so that a poor default configuration does not unfairly eliminate a strong plan.
34. As a performance engineer, I want only the winning candidate to receive full kernel-parameter autotuning, so that plan search and configuration search remain separate and tractable.
35. As a compiler user, I want plan selection based on end-to-end runtime execution, so that launch and materialization costs are not hidden behind a fast isolated kernel.
36. As a compiler user, I want compilation time excluded from the runtime performance objective, so that the fastest generated operator wins.
37. As a performance engineer, I want representative workloads normalized before aggregation, so that long Sequences do not dominate merely because their absolute latency is larger.
38. As a performance engineer, I want candidate scores computed as an equal-weight geometric mean of per-workload latency ratios, so that short, medium, and long workloads contribute comparably.
39. As a performance engineer, I want a candidate that fails any required representative workload removed, so that incomplete evidence cannot win.
40. As a compiler maintainer, I want representative workloads owned and versioned by the compiler, so that users do not configure concerns outside Algorithm IR.
41. As a dense execution user, I want the tuning policy to cover multiple Batch and Sequence scales, so that the fixed kernel is not selected from one narrow shape.
42. As a packed execution user, I want the tuning policy to cover uniform, long-tail, and empty-containing distributions, so that segmentation behavior is represented.
43. As a compiler user, I want `autotune=False` to perform no new measurements, so that compile behavior is predictable.
44. As a compiler user, I want `autotune=False` to consume matching successful tuning records, so that previously paid tuning cost is not discarded.
45. As a compiler user, I want `autotune=True` to tune forward and generated backward kernels, so that one option controls all requested kernel configuration search.
46. As a compiler user, I want `optimized` and `autotune` to be independent, so that generic kernels can be tuned and specialized plans can be selected from cached or static evidence.
47. As a compiler user, I want `optimized=False` to retain validation, canonicalization, safe simplification, and generic lowering, so that it disables only structural optimization search.
48. As a compiler user, I want `optimized=False, backward=True` to generate generic forward and backward kernels, so that disabling optimization does not remove gradients.
49. As a compiler user, I want compile-time arguments to describe metadata rather than carry actual Tensor values, so that compilation does not depend on example allocations.
50. As a compiler user, I want Head and Feature role sizes supplied once by logical role, so that input shapes and roles are not redundantly declared.
51. As a compiler user, I want Algorithm IR dtypes to remain authoritative, so that compile-time overrides cannot change numerical semantics.
52. As a deployment user, I want the Target supplied explicitly to compilation, so that code generation never depends on process-global current-device state.
53. As a compiler user, I want dense and packed mode fixed at compilation, so that a compiled operator does not dynamically switch physical execution contracts.
54. As a dense execution user, I want Batch size to remain dynamic, so that one compiled operator serves different batch workloads.
55. As a dense execution user, I want Sequence length to remain dynamic, so that one compiled operator serves arbitrary positive uniform lengths.
56. As a packed execution user, I want logical-sequence count to remain dynamic, so that offsets shape does not force recompilation.
57. As a packed execution user, I want total-token count and offsets values to remain dynamic, so that different length distributions reuse one artifact.
58. As a packed execution user, I want all-empty invocations to remain valid, so that dynamic packed execution preserves identity State semantics.
59. As a compiler user, I want runtime Tensors checked against fixed Head, Feature, dtype, layout, mode, and Target contracts, so that dynamic workload support does not weaken validation.
60. As a compiler user, I want runtime Batch and Sequence metadata excluded from binary identity, so that dynamic invocations reuse the same kernel.
61. As a kernel author, I want launch dimensions and workspace sizes computed from runtime metadata without selecting another plan, so that dynamic workloads remain compatible with fixed compilation.
62. As a runtime user, I want workspace obtained through a compiler-controlled reusable allocator, so that kernels do not perform uncontrolled device allocation.
63. As a runtime user, I want an explicit resource error when a dynamic workload exceeds the selected implementation's envelope, so that the compiler does not silently recompile or switch plans.
64. As a framework user, I want compilation to return one callable forward operator, so that invocation remains simple.
65. As an inference user, I want `backward=False` to generate no backward kernel or compiler backward graph, so that inference avoids unnecessary work and context.
66. As a framework user, I want grad-enabled invocation rejected when backward was not compiled, so that no eager or implicit fallback changes execution semantics.
67. As a framework user, I want `backward=True` to generate a compiler-derived backward kernel and connect it through PyTorch autograd, so that ordinary `.backward()` invokes generated code.
68. As a compiler maintainer, I want PyTorch autograd to transport cotangents and context only, so that compiler VJPs remain the sole backward semantic source.
69. As a compiler maintainer, I want forward and backward Mathematical Plans selected independently, so that each direction can use its fastest valid structure.
70. As a compiler maintainer, I want forward and backward lowerers selected independently, so that implementation reuse does not require symmetric kernels.
71. As a compiler maintainer, I want forward and backward kernel parameters tuned independently, so that each direction uses an appropriate configuration.
72. As a correctness reviewer, I want forward candidates proven equivalent to canonical forward semantics, so that specialization cannot change outputs or State.
73. As a correctness reviewer, I want backward candidates proven equivalent to canonical Backward IR, so that independent lowering cannot change gradients.
74. As a compiler maintainer, I want a named context contract between forward and backward candidates, so that compatibility is checked before code generation is published.
75. As a backward-kernel author, I want to declare named values required from forward, so that data dependencies are explicit.
76. As a backward-kernel author, I want to declare values that the backward implementation can recompute, so that save-versus-recompute strategies can compete.
77. As a forward-kernel author, I want to declare named context values produced by forward, so that compatible backward candidates can be discovered.
78. As a compiler maintainer, I want context names to describe semantic values such as token State or affine prefix, so that contracts do not depend on temporary-buffer positions.
79. As a compiler maintainer, I want context schemas to include dtype, role, and shape rules, so that equal names cannot connect incompatible Tensors.
80. As a performance engineer, I want context allowed to contain inputs, outputs, token States, chunk States, summaries, and gates, so that performance is not constrained by an artificially narrow ABI.
81. As a performance engineer, I want sequence-length-dependent context allowed when resources permit, so that saving can outperform recomputation.
82. As a performance engineer, I want context memory treated as a hard feasibility constraint rather than the primary ranking objective, so that latency remains the optimization target.
83. As a compiler maintainer, I want context hidden inside the autograd wrapper, so that users do not manage a backward ABI.
84. As a compiler maintainer, I want fixed prepare kernels permitted to materialize context not produced by forward, so that forward binaries can sometimes be reused.
85. As a compiler maintainer, I want prepare kernels treated as explicit cached artifacts, so that context preparation is not a runtime fallback.
86. As a compiler maintainer, I want forward/backward pairs rejected when required context is neither produced nor recomputable, so that incompatibility never appears during invocation.
87. As a framework user, I want unused output cotangents treated as mathematical zero, so that outputs not participating in a loss contribute no gradient.
88. As a framework user, I want an unused final State cotangent treated as mathematical zero, so that output-only losses require no synthetic Tensor.
89. As a streaming user, I want gradients to an explicitly provided initial State propagated through autograd, so that continuation can participate in differentiation.
90. As an inference user, I want an omitted compiler-created zero State to create no user-visible initial-State gradient, so that only actual inputs receive gradients.
91. As a cache user, I want semantic, plan, kernel, tuning, context, and binary identities separated, so that each artifact is reused only under its real invariants.
92. As a cache user, I want tuning identity isolated by exact H20 product, software stack, resource fingerprint, and clock policy, so that incompatible measurements are not reused.
93. As a cache user, I want the representative-workload policy version included in tuning identity, so that policy changes invalidate stale measurements.
94. As a cache user, I want forward-only and backward-enabled compilations to share identical forward binaries when context adds no kernel work, so that duplicate artifacts are avoided.
95. As a cache user, I want a forward artifact that produces extra backward context to have a distinct identity, so that inference does not inherit hidden work.
96. As a cache user, I want prepare and backward artifacts independently identified and atomically published, so that partial compositions are never observed.
97. As a cache user, I want only successful measurements and binaries persisted, so that transient failures do not poison future compilation.
98. As a compiler user, I want specialized candidate failures summarized once with candidate and reason, so that optimization gaps are visible without warning spam.
99. As a compiler user, I want compilation to continue after a specialized candidate fails, so that another specialized or generic implementation can succeed.
100. As a compiler user, I want generic baseline failure to raise a structured StatefulCompilationError, so that compilation never returns an unusable operator.
101. As a maintainer, I want ordinary AttentionEngine and LinearAttentionEngine behavior unchanged, so that this design remains scoped to Stateful Operator.
102. As a reviewer, I want implementation and performance evidence gathered on an otherwise idle H20, so that plan and tuning conclusions reflect the production Target.

## Implementation Decisions

- Preserve the existing finite Algorithm IR and define compatibility as acceptance of previously unseen compositions within that language. Arbitrary frontend IR recognition is not part of this work.
- Keep Scanability, Target capability, and optimization matching as separate decisions. A Target-supported Scanable Program always receives a generic dense affine candidate.
- Run deterministic validation and typing before canonicalization, simplification, planning, or lowering.
- Maintain a Canonical IR that removes representation-only differences while preserving ordered Feature roles, authored noncommutative Propagation order, and floating-point association.
- Add one mandatory safe-simplification phase. Its registry contains only rules that are unconditionally valid under the Algorithm IR dtype, ordering, and VJP semantics. It does not make performance-dependent choices.
- Represent the mandatory simplification output as Simplified IR. This is an internal optimization view and does not replace public Algorithm IR identity.
- Add a Mathematical Planner that enumerates equivalent Mathematical Plans from one Simplified IR. Each plan records its representation, applied rewrite identities, semantic regions, preconditions, and conflicts.
- Treat scale fusion, diagonal preservation, Product Injection/GEMM recognition, Rank-One factorization, shared-readout fusion, and dense affine representation as model-independent planner concerns.
- Require optional planner rewrites to preserve the compiler's numerical contract and canonical VJP semantics. Mathematical real-number equivalence alone is insufficient when it would reassociate floating-point expressions.
- Define Mathematical Plan as algorithm structure only. It excludes kernel names, target schedules, launch counts, tile sizes, warps, stages, benchmark measurements, and tuning results.
- Add a registry of kernel lowerers. A lowerer declares which Mathematical Plans and concrete specializations it supports, estimates feasibility and cost, and emits a Lowered Candidate without changing the plan's mathematics.
- Permit multiple lowerers for the same Mathematical Plan. A new implementation of an existing mathematical structure should normally require only a new lowerer registration.
- Require new mathematical representations to be added through planner rules, not hidden inside a lowerer.
- Represent backend implementation choices as Lowered Candidates. Candidate identity includes its Mathematical Plan, lowerer, direction, context contract, and resource model, but not its final tuned kernel parameters.
- Represent TileLang scheduling parameters separately as KernelConfig. The selected Lowered Candidate is fixed before full KernelConfig autotuning.
- Always include generic dense affine forward and, when requested, generic compiler-derived backward candidates.
- When optimization is enabled, statically validate and estimate every Lowered Candidate, retain the strongest three, and add the generic baseline if it is not already present.
- Give each measured candidate a small, fixed parameter-probe budget before cross-plan comparison. Give only the winning candidate the full configuration-search budget.
- Use compiler-owned, versioned representative workload policies. Dense policies cover multiple Batch and Sequence scales. Packed policies cover uniform, long-tail, and empty-containing distributions. Exact v1 points are frozen only after H20 pre-experiments.
- Rank measured candidates by the equal-weight geometric mean of per-workload latency divided by the best valid latency for that workload. A candidate that fails a required workload is removed.
- Exclude compilation time from plan and configuration performance objectives. Record it separately if useful.
- Treat device memory, context memory, workspace, register pressure, shared memory, and other Target resources as hard feasibility constraints. Rank feasible candidates primarily by runtime latency.
- Replace tensor-based compilation with metadata-only compilation. Do not retain a compatibility shim that accepts example Tensors.
- Accept fixed sizes keyed by HeadRole and FeatureRole. Algorithm IR remains the sole source for input names, role tuples, dtypes, differentiability, State roles, readout dtypes, and Head Mapping.
- Reject Batch and Sequence entries in compile-time role sizes. Batch and Sequence are always runtime workload dimensions.
- Require the compile call to name an explicit Target and one physical execution mode, dense or packed. Never derive Target from a process-global current device.
- Expose independent compile choices for structural optimization, autotuning, and backward generation. Structural optimization controls Mathematical Plan search; autotuning controls creation of new performance measurements and KernelConfig search; backward controls generation of compiler backward artifacts and autograd integration.
- Permit matching successful tuning records to be consumed when autotuning is disabled. Do not create new measurements in that mode.
- Keep Head and Feature dimensions, dtype, canonical layout, execution mode, Target, chosen plan, lowerer, KernelConfig, and context contract fixed in a compiled artifact.
- Keep dense Batch, dense Sequence length, packed logical-sequence count, packed total-token count, packed offsets shape and values, actual Tensor values, and runtime workspace size dynamic.
- Derive dense Batch and Sequence length from invocation Tensor shapes. Do not add a redundant runtime sequence-length argument.
- Derive packed logical-sequence count from offsets length and total-token count from input shape. Keep offsets rank, int32 dtype, device, contiguity, start/end, and monotonicity requirements as runtime validation contracts.
- Generate fixed implementations whose launch bounds and workspace sizes are computed from runtime workload metadata. Runtime metadata must not trigger planning, lowering, autotuning, kernel-family dispatch, or compilation.
- Use a compiler-controlled reusable workspace allocation path. Device-side dynamic allocation is not permitted.
- Return one callable compiled forward operator. Invocation continues to produce immutable ExecutionResult.
- When backward is disabled, generate no backward kernel and no compiler backward graph. Reject grad-enabled invocation rather than falling back to eager or framework-derived backward.
- When backward is enabled, derive canonical Backward IR exclusively from compiler-owned VJPs, simplify and plan it, lower it independently, and connect the selected backward implementation through PyTorch autograd.
- Use PyTorch autograd only as the transport for saved Tensors and output/final-State cotangents. It does not derive the backward algorithm, select kernels, or provide a semantic fallback.
- Allow forward and backward to select different Mathematical Plans, lowerers, and KernelConfigs, provided each implementation is equivalent to its canonical IR and their context contracts are compatible.
- Introduce a compiler-owned named context schema. Each context value has a semantic structured name plus dtype, logical roles, and shape rules.
- Allow context to contain any compiler-defined Tensor needed for performance, including original inputs, outputs, token States, chunk States, affine summaries, and gates.
- Store runtime context internally through the generated autograd wrapper. Do not return context in ExecutionResult or expose a public backward handle.
- Require forward candidates to declare context they produce and backward candidates to declare context they require and can recompute. A pair is valid only when every required value is produced or explicitly recomputable.
- Permit fixed prepare kernels to materialize named context values not produced by the selected forward kernel. Prepare kernels are explicit lowered and cached artifacts, not runtime fallbacks.
- Permit sequence-length-dependent context when it passes resource constraints. Saving, preparing, and recomputing are performance alternatives rather than public API choices.
- Preserve existing cotangent semantics: an absent output or final-State cotangent is mathematical zero. Gradients to an explicitly supplied initial State flow through autograd; a compiler-created zero State has no user Tensor to receive such a gradient.
- Separate semantic identity, Mathematical Plan identity, Lowered Candidate identity, KernelConfig identity, tuning identity, context-contract identity, and binary artifact identity.
- Exclude dynamic Batch, Sequence, packed counts, packed offsets shape, and offsets values from binary identity.
- Include exact Target product/environment, software stack, resource fingerprint, clock policy, and representative-workload policy version in tuning identity.
- Allow forward-only and backward-enabled compilation to share a forward binary only when the binary and context-production behavior are identical. Additional context materialization creates a distinct forward artifact.
- Cache plan measurements, kernel tuning results, forward binaries, prepare binaries, backward binaries, and autograd compositions under their actual contracts. Publish artifacts atomically with process-safe locking.
- Persist successful results only. Do not create persistent negative cache entries for candidate compilation, driver, resource, or measurement failures.
- Aggregate all failed specialized attempts into one warning that names each candidate, failure stage, exception type, concise reason, and final selected candidate. Warning text is diagnostic rather than a stable API.
- Continue after specialized failures. Raise a structured StatefulCompilationError if the generic baseline cannot compile or execute its required validation workload.
- Implement the change as a clean cutover across repository callers, tests, examples, and API documentation. Do not preserve tensor-based compile aliases, deprecated option names, or lazy invocation compilation.
- Develop and validate the implementation on CUDA H20. The present specification is a design artifact; concrete workload values, performance results, and resource limits require later H20 experiments.

## Testing Decisions

- Use one primary public behavioral seam: construct StatefulOperator from explicit Algorithm IR, compile it from role sizes, Target, mode, and compile options, then invoke the returned callable with real keyword-bound Tensors and consume ExecutionResult or autograd gradients.
- Prefer this compile-and-invoke seam over direct tests of simplifier, planner, lowerer, selector, cache, or context internals. Internal seams may be tested only where a public observation cannot isolate deterministic algebraic legality or identity behavior.
- Test observable mathematical behavior against independent PyTorch recurrences. A compiler implementation must not serve as its own reference.
- Test that previously unseen legal compositions compile through generic dense affine semantics without adding a whole-program pattern.
- Test mandatory simplification through observable canonical identity and numerical behavior: commutative ordering, identity removal, empty Propagation, safe zero/one elimination, preserved association, ordered noncommutative Propagations, and ordered Feature roles.
- Test planner rewrites with asymmetric, non-square, noncommuting examples so that incorrect direction, order, fusion, or factorization is observable in outputs and gradients.
- Test that specialized and generic candidates produce equivalent forward outputs, final State, input gradients, and initial-State gradients within the declared dtype tolerance.
- Test that adding or selecting an optimization never changes Scanable/Recurrent classification or Target-capability diagnostics.
- Test `optimized=False` through the public seam and verify it remains correct for forward-only and backward-enabled compilation.
- Test `autotune=False` without pre-existing records and verify no measurement is produced; then install a matching successful record and verify it is consumed without a new benchmark.
- Test candidate selection with deterministic fake timing at the narrow selector seam where real H20 measurement would be nondeterministic. Verify Top 3 retention, forced generic inclusion, normalized geometric-mean ranking, failed-candidate removal, and deterministic tie-breaking.
- Test candidate failures through compilation: one or more specialized failures must produce one aggregate warning and a valid selected operator; generic failure must produce StatefulCompilationError.
- Test the warning for candidate identity, stage, exception type, concise reason, and selected fallback without freezing incidental traceback or complete message formatting.
- Test metadata-only compilation and reject actual Tensor arguments, missing fixed roles, unknown roles, Batch/Sequence role sizes, unsupported Feature sizes, invalid Head Mapping sizes, missing Target, and invalid mode.
- Reuse one dense compiled operator across multiple Batch sizes and Sequence lengths, including `1`, `63`, `64`, `65`, `127`, `129`, and representative larger lengths. Verify output, State, gradients, and inactive-tail identity.
- Reuse one packed compiled operator across multiple logical-sequence counts, total-token counts, offsets shapes, uniform distributions, long-tail distributions, empty interior/leading/trailing sequences, and all-empty invocations.
- Verify dynamic workload dimensions do not change semantic or binary artifact identity. Verify fixed Head, Feature, dtype, layout, mode, Target, plan, lowerer, configuration, or context-contract changes do.
- Test runtime validation for mismatched Batch/Sequence axes, fixed role sizes, dtype, device, layout, mode, State shape, offsets rank/dtype/device/contiguity/start/end/monotonicity, and total-token agreement.
- Exercise runtime workloads large enough to exceed a configured test resource envelope and verify a structured resource error without replanning, recompilation, or fallback.
- Test `backward=False` with inference inputs and verify normal execution. Test grad-enabled invocation and verify an explicit error rather than eager autograd fallback.
- Test `backward=True` through ordinary PyTorch `.backward()`. Verify output-only, final-State-only, and joint cotangent paths; missing cotangents; all differentiable inputs; grouped Head reductions; aliased input-name accumulation; and explicit initial-State gradients.
- Test context compatibility at the narrow internal seam with named schemas: produced, required, recomputable, missing, dtype-mismatched, role-mismatched, and shape-rule-mismatched values.
- Test at the public seam that a selected compatible forward/backward pair produces correct gradients whether context is saved by forward, produced by a fixed prepare kernel, or recomputed by backward.
- Test context resource filtering with token-level context and ensure an infeasible candidate is removed before publication while a feasible alternative succeeds.
- Test that forward-only and backward-enabled compilation share a forward artifact when context is wrapper-saved from existing inputs/outputs, and do not share when forward must materialize additional context.
- Test plan, tuning, forward, prepare, backward, and composition caches for successful reuse, identity isolation, process-safe locking, and atomic publication.
- Test that transient compile, driver, resource, and measurement failures leave no persistent negative records or partial artifacts.
- Use existing Stateful Operator unit and functional tests as prior art for Algorithm IR construction, ExecutionResult assertions, continuation, packed offsets, structured errors, and cache identity.
- Use existing AttentionEngine TileLang autograd wrappers as prior art for saving forward context and invoking generated backward kernels, while retaining compiler-derived Backward IR as the Stateful Operator semantic source.
- On an otherwise idle H20, freeze the representative-workload policy, software stack, clocks, warmup, repetitions, and measurement order before evaluating selection quality.
- On H20, verify that small parameter probes do not systematically eliminate the final fastest Mathematical Plan; revise the fixed probe policy before acceptance if they do.
- On H20, measure the selected fixed implementation across all dense and packed policy workloads and confirm runtime Batch/Sequence changes do not trigger code generation or kernel-family dispatch.
- On H20, benchmark generic and selected specialized implementations in interleaved batches and retain the existing statistical latency and peak-memory acceptance gates where an old implementation baseline exists.
- Run the CPU-safe semantic, identity, cache, and error suite first; then focused H20 lowering/autograd tests; then packed/dynamic workload tests; then performance selection evidence; and finally the existing functional GPU suite.

## Out of Scope

- Recognizing arbitrary PyTorch FX, TorchScript, MLIR, ONNX, backend, or other external IR directly.
- Adding model-name, frontend-name, input-name, or whole-profile dispatch.
- Adding public model-specific Stateful Operator factories or dedicated GDN execution.
- Changing ordinary AttentionEngine, decode attention, MLA, Online Functions, score modifiers, mask modifiers, CustomIO, or LinearAttentionEngine behavior.
- Expanding the public Algorithm IR node vocabulary, State count, State rank, Head Mapping kinds, dtype set, or H20 Feature capability.
- Supporting CPU, ROCm, H100, non-H20 CUDA products, or unidentified MIG devices as production Stateful Operator Targets.
- Supporting arbitrary-stride layouts, implicit copies, implicit transposes, hidden dtype conversion, device moves, or padding-based packed execution.
- Permitting floating-point reassociation, approximate algebraic rewrites, or relaxed mathematical semantics solely for performance.
- Using eager token recurrence, token-serial execution, PyTorch eager backward, or framework-derived autograd as a production semantic fallback.
- Exposing a public compiler-analysis report, Mathematical Plan selector, KernelLowerer registry, KernelConfig API, context object, raw backward operator, or tuning-workload configuration.
- Allowing users to specify runtime Batch/Sequence maxima, sequence-length buckets, workload weights, candidate counts, or plan-selection objectives.
- Runtime selection among multiple plans, lowerers, or kernel configurations based on Batch, Sequence length, offsets, or Tensor values.
- Compiling separate kernels for short/medium/long runtime buckets or dispatching among them at invocation time.
- Treating compilation latency as a plan-selection or merge-performance objective.
- Persisting failed-candidate or negative cache entries.
- Defining concrete v1 tuning workload points without H20 pre-experimental evidence.
- Implementing or claiming H20 results as part of this discussion-only specification.

## Further Notes

- The public testing seam is metadata-only `StatefulOperator.compile(...)` followed by invocation of the returned callable and, when requested, ordinary PyTorch `.backward()`. This is the highest existing seam that observes compile contracts, dynamic runtime workload behavior, forward semantics, State continuation, generated backward, diagnostics, and cache reuse together.
- Mathematical Plan, Lowered Candidate, and KernelConfig are deliberately separate. Plan answers which equivalent algorithm structure is used; candidate answers how a backend implements it; configuration answers how the selected implementation is scheduled.
- “Fixed kernel” means compilation fixes the implementation and its component kernels. A compiled backward-capable operator may contain fixed forward, prepare, backward-preparation, backward, and postprocessing kernels. Runtime does not select or generate them.
- A prepare kernel regenerates or materializes named backward context not emitted by the selected forward kernel. It is useful when reusing a fast forward binary is cheaper than changing forward to save additional intermediates.
- Latency-first does not mean memory-blind. Memory and resource limits eliminate candidates before ranking; memory is not otherwise weighted against latency in the first selector.
- The current implementation already contains partial structural analysis, generic affine and elementwise scans, scalar-factorized recognition, local autograd Functions, and atomic plan-manifest publication. These are migration inputs, not evidence that the full design is complete.
- Existing documentation currently describes tensor-based fixed specialization and therefore must be updated during implementation. This specification intentionally requires a clean metadata-only compile cutover.
- ADR 0001 remains authoritative on compositional lowering, generic dense affine semantics, compiler-derived Backward IR, and prohibition of model-specific execution. A later ADR update is warranted only after H20 evidence resolves the hard-to-reverse planner/lowerer/cache architecture and the existing ADR acceptance conditions.
- Development and performance validation will occur on H20 hardware later. Until then, workload values, resource envelopes, selector fidelity, and performance claims remain proposed rather than demonstrated.
