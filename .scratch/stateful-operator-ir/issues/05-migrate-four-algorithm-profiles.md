# Migrate Four Algorithm Profiles

Labels: `ready-for-agent`

## Goal
Express four realistic recurrent profiles solely as explicit compositions of the generic public IR.

## Scope
- Author GLA, RetNet recurrent, selective diagonal/Mamba2-style state evolution, and GDN profiles.
- Use typed roles, Axis Scale, generic Rank-One Propagation, Product Injection tuples, State Contractions, and explicit Head Mapping.
- Add independent mathematical references for dense/packed forward, continuation, and gradients.
- Keep profile names in examples/tests only, never compiler dispatch.

## Non-goals
No public model factory, model-name selector, whole-profile optimizer, legacy node compatibility, or backend shortcut.

## Dependencies
Issues 01–04.

## Acceptance
Every profile crosses `StatefulOperator → ExecutionResult`, supports declared initial/final State and gradient modes, and matches its independent reference. Mamba2 equations and dtype behavior are explicit. Adding the profiles adds no compiler branch keyed by model, frontend, or input names.
