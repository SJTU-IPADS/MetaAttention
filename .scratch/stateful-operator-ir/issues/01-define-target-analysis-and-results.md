# Define Target Analysis and Public Results

Labels: `ready-for-agent`

## Goal
Implement the internal, code-generation-free analysis and the stable public invocation contract for the compositional Stateful Operator compiler.

## Scope
- Validate the first schema: one named matrix State with two ordered Feature roles.
- Canonicalize typed expressions, ordered propagation, Product Injections, Head Mapping, and named State Contractions.
- Resolve optional Target constraints and deterministic validation order.
- Implement immutable `ExecutionResult`, `NamedOutputTuple`, and `StateTuple`.
- Expose structured `StatefulCompilationError` only; keep analysis reports private.

## Non-goals
No backend lowering, model-name dispatch, public `analyze_program`, GDN adapter, or `LinearAttentionEngine` migration.

## Dependencies
None. This defines the public seam consumed by all later issues.

## Acceptance
Through `StatefulOperator(AlgorithmIR)` and real keyword-bound tensors, valid programs return immutable named outputs and optional named final State. Invalid programs expose stable category codes, canonical IR paths, and JSON-safe details. Single output/state values are never unwrapped. Target/resource errors remain distinct from Recurrent classification.
