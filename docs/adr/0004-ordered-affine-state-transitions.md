---
status: accepted
---

# Ordered Affine Steps and Coupled State Transitions

## Decision

Schema v4 extends State Transitions with ordered affine steps and explicit
Prior/Current State references. The legacy grouped spelling keeps its existing
semantics. State Terms remain a Readout expression; arbitrary State-dependent
Transition expressions are not admitted.

`StateTransition(state, steps=(...))` applies each step to the target's running
value. Steps may be AxisScale, RankOnePropagation, ProductInjection or
StateInjection; Identity is omitted. Grouped `propagations/injections` cannot
be supplied with `steps`. Their existing tuples are available as projections
for structural analysis, while `ordered_steps` is the semantic source.

`StateReference(name, prior=True)` selects the State before this token;
`StateReference(name)` selects that State after its complete current-token
Transition. Current references form a Transition dependency DAG, with a stable
topological schedule. A cycle, including a current self reference, fails with
`IR_DEPENDENCY`. Earlier steps within the same target affect its running value;
there is no implicit partial update read or arbitrary named intermediate version.

`StateInjection(source, coefficient)` adds `coefficient * source`.
The rank-one form adds `coefficient * left * contract(right, source)` along
its declared Feature role. Both forms are additive, without an identity action.
Coefficients and vectors must be Input-driven. Source and target roles must
match, including Head role and Feature order.

The first coupled lowering accepts compatible States whose non-scalar actions
all use one shared Feature axis. It stacks States on that axis, substitutes
completed source affine maps for current references and identity block rows
for prior references, and obtains one joint per-token affine map. Its summary
size is independent of sequence length at fixed State dimensions. A segmented
parallel scan executes it; there is no sequential token fallback. Both matrix
axis orientations and vector States are supported. Mixed-axis or heterogeneous
coupling is rejected with a structured type error.

Ordered independent State Transitions compose affine steps, transforming the
bias as well as the propagated State. Where the factored matrix proof applies,
each write's factors are transformed through later actions before entering the
existing compiled kernel. This avoids exponential outer-product expansion;
with n writes the public IR is O(n), and factor preparation can be O(n^2).
This algebraic rewrite changes FP32 association, so numerical acceptance is
based on an explicit error budget rather than bitwise identity. Existing
weighted factor ordering remains preserved inside each outer write.

## Considered Options

Simply permitting StateTerm in arbitrary Transition expressions would admit
nonlinear feedback without an affine proof. A model-specific Momentum kernel
would bypass compositional typing and summary semantics. Flattening all matrix
States into a dense transform over their full element count would unnecessarily
square K*V for Momentum. The selected block-axis representation reuses shared
affine scan machinery and charges workspace against the joint axis.

Implementing a general Scan Stage DAG here would mix two distinct proofs:
Momentum is jointly affine, while preconditioners transform auxiliary scan
trajectories nonlinearly before the main scan. That capability remains a later
extension.

## Consequences and Validation

[Momentum Delta Rule](../../examples/momentum_delta_rule.py) uses two matrix
States and a current Momentum → Memory dependency. It follows FLA's
`p^T S_prior - v` sign convention and exposes `use_p_times_alpha`; the public
State order is `(memory, momentum)`, corresponding to FLA's `(S, M)` stack.
[Delta Product](../../examples/gated_delta_product.py) now authors interleaved
writes directly and retains factored dense/packed and backward selection.

Tests at the public compile/invoke seam cover independent recurrence parity,
nonzero initial/final States, continuation, grouped heads, unequal Feature
sizes, matrix-axis orientations, vector States, prior Readouts, packed empty
sequences, and output/final/combined gradient losses. Cycles and State-derived
coefficients are rejected. GPU checks validate H20 correctness and the retained
Delta Product compiled paths; they are not performance benchmarks.

The schema identity changes to `stateful-operator-ir/4`. Step order and source
State versions enter cache identity. Joint-axis workspace is estimated before
execution. Coupled execution uses the differentiable reference scan; a
specialized joint kernel and nonlinear scan stages are still future work.
