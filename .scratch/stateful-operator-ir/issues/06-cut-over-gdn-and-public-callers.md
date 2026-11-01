# Cut Over GDN and Public Callers

Labels: `ready-for-agent`

## Goal
Complete the strict public cutover to the compositional Stateful Operator language.

## Scope
- Remove `GDNEngine`, its exports, implementation, tests, docs, and callers.
- Remove obsolete `RankOneDelta`, `OuterProduct`, `MatrixReadout`, dictionary-State, and tuple-result call patterns from Stateful Operator examples and tests.
- Route GDN examples through generic Rank-One Propagation and other generic nodes.
- Preserve `LinearAttentionEngine` interface and behavior outside this guarantee.

## Non-goals
No compatibility shims, deprecated aliases, GDN-specific execution, or migration of `LinearAttentionEngine`.

## Dependencies
Issue 05 profiles.

## Acceptance
Repository callers use the new `ExecutionResult`/`StateTuple` seam. No public or internal GDN execution bypass remains. GDN behavior still matches its independent reference through generic lowering. Legacy `LinearAttentionEngine` tests remain unchanged and passing.
