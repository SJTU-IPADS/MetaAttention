# 24: Compose two scaled axes elementwise

**What to build:** Let a Program whose Transition scales both State Features compose the two per-axis factors without falling back to the dense diagonal.

**Blocked by:** 20/Lower per-axis diagonal decay elementwise.

**Status:** complete

`_dense_summary_required` still returns True when two different Feature roles carry a single-role Axis Scale, because the prefix kernel reads one factor per matrix element and a row factor times a column factor needs both. That is the shape of `gated_delta_rule`'s `left_gate`/`right_gate` profile and of `fla.ops.gla.fused_recurrent_gla(gk, gv)`, and it costs the same `[T,H,K,K]`/`[T,H,V,V]` materialization that 20 removed for a single axis.

- [x] The elementwise prefix kernel takes a second factor slot, or lowers the pair without materializing a `[T,H,R,C]` tensor.
- [x] The elementwise VJP returns one gradient per factor, reducing each along the axis the other factor spans, and still matches the FP32 recurrence for both gates.
- [x] The chunk carry composes the pair across chunk boundaries in dense and packed modes.
- [x] `gated_delta_rule(left_gate, right_gate)` and the two-Axis Scale GLA profile select a non-dense plan and stay within 2x of the single-axis cost for the same State shape.
- [x] `_dense_summary_required` no longer treats two scaled axes as dense, and the guard in `_axis_scale_summary` that rejects the composition is removed with it.

Delivered: the prefix kernel takes one factor per State axis (`scale` as `[T,H,R,1]` and `second` as `[T,H,1,C]`) and composes them per element while it walks the tokens, writing the two cumulative products instead of one, so no `[T,H,R,C]` factor is ever materialized. `_axis_scale_summary` returns the pair rather than raising, `_dense_summary_required` only forces dense factors for a rank-one propagation, and the runtime follows the compiled `dense_summary` decision instead of recomputing the rule.

The VJP follows from the same reverse walks that already serve one factor: the per-element gradient is taken with respect to the product `r * c`, so each factor's gradient is that elementwise gradient contracted with the other factor along the axis it spans. No division by a factor appears, so a vanishing gate is not amplified. The autograd transport composes the pair before the walks and returns one gradient per factor, and a compilation that asks for a backward now selects the same non-dense plan.

Measured with a compiled backward the pair also holds the single-axis per-token workspace: 384 KiB/token against 640 KiB/token for the dense factors, and 2.411 ms against 1.856 ms for one scaled axis at T=2048.

Measured at T=2048, H=8, K=V=64, inference, idle H20, with the new `dual_gate_gla_attention` profile (FLA `fused_recurrent_gla(gk, gv)` shape):

| Program | latency | lowerer |
| --- | --- | --- |
| `gla` (one scaled axis) | 1.870 ms | `tilelang-elementwise` |
| two scaled axes, elementwise pair | 2.432 ms | `tilelang-elementwise` |
| two scaled axes, pair with a compiled backward | 2.411 ms | `tilelang-elementwise` |
| two scaled axes, dense summary | 80.545 ms | dense factors |

The pair is 1.30x the single-axis cost and 33x faster than the dense summary, which also cannot run at T=2048 under the 1 GiB envelope while the pair can. Tests: `tests/functional/test_two_axis_elementwise.py` covers dense parity against an FP32 token recurrence, packed parity with leading and trailing empty sequences, and the backward-compilation gate.

Tests: `tests/functional/test_two_axis_elementwise.py` covers dense parity against an FP32 token recurrence, packed parity with leading and trailing empty sequences, and the paired VJP's gradients against the same oracle; `tests/unit/test_stateful_selection.py` pins the analysis rule for both a forward-only and a backward-enabled compilation.

**Review round.** A two-axis review of the implementation commits found three things worth recording.

1. The first factor-composition rewrite dropped an accumulated whole-State scale when a per-axis scale followed it (`single` was cleared instead of folded in), so a Program carrying both would have silently lost the scalar decay. Fixed: the accumulated whole-State factor folds into the first per-axis factor that appears, and `tests/unit/test_two_axis_summary.py` pins the elementwise product `w * r * c` for both authored orders and for the three-factor case.
2. Making two axes non-dense also left the generic family's backward unsatisfied: with `dense_summary` false its forward composes an elementwise factor, but `backward_context_family` still asked for dense `left`/`right` factors, which surfaces as a missing-context error at invocation. Fixed: that branch now follows `dense_summary`, and the same test file compiles a generic two-axis Program with a backward and asserts the bound family is the elementwise one.
3. The paired update branches in the prefix kernel repeated the same accumulation shape; they now share one path through a single composed factor, with `accumulated_scale` still tracking the row factor whose shape the kernel writes.

One reported finding (a [R,C]/[R,1] expansion failure in the paired VJP) was retracted by the reviewer after re-reading the composition order, and my own tests covering that path agree it does not reproduce.

**Open spec-conformance item (standards axis, P1, unchanged by this ticket).** `docs/stateful-operator-ir-spec.md` Implementation Decisions says the compiler performs no implicit `.contiguous()` outside declared seams, and the factor materialization in `_lower_parallel_scan` calls it unconditionally for both factor slots. It is pre-existing for the single-factor slot; this ticket adds the same call for the second. Measured: replacing the call with a contiguity guard fails three real cases -- `test_fla_coverage.py::test_coverage_factory_compiles_and_runs_on_cpu[retention_attention]` and two cases in `test_two_axis_elementwise.py` -- because the summary factors are derived from the token-major input views and inherit their strides, so the materialization is load-bearing rather than defensive. Remediation, in preference order: have `_normalize_inputs` publish canonical contiguous factors at one declared seam, or reject non-contiguous factors where they enter the summary (spec section "Reject noncontiguous tensors before cache lookup/codegen"), which would make the call site a no-op and delete the copy. Both are wider than this ticket and are left open with the failing guard recorded here rather than shipped as a regression.
