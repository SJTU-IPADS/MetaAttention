# Lower Derived Backward and State Continuation

Labels: `ready-for-agent`

## Goal
Lower compiler-derived Backward IR and complete explicit State continuation semantics.

## Scope
- Generate reverse chunk composition and per-node VJPs from canonical Backward IR.
- Support output-only, final-State-only, and joint cotangents.
- Accept optional immutable `StateTuple`; create zero State when omitted.
- Keep State algebra/cotangents FP32 and cast only at invocation entry/final materialization.
- Accumulate identical-Tensor aliases and reject overlapping distinct storage views.

## Non-goals
No autograd recurrence backend, hand-written profile backward, mutable operator-owned State, or BF16 State.

## Dependencies
Issues 01–03.

## Acceptance
Through `StatefulOperator`, forward/backward and prefix/suffix continuation match independent references. Gradients cover every differentiable binding and provided initial State. Unrequested final State is absent and unmaterialized. Identical Tensor bindings accumulate contributions; unsupported aliasing fails structurally.
