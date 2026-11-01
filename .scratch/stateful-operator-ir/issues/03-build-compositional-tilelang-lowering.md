# Build Compositional TileLang Lowering

Labels: `ready-for-agent`

## Goal
Lower every H20-supported Scanable Program through one parallel chunked forward compiler.

## Scope
- Implement generic dense affine chunk summaries, associative composition, and state application.
- Generate semantics from Axis Scale, Rank-One Propagation, Product Injection, Head Mapping, and State Contraction nodes.
- Support multiple injections and readouts, unequal State Feature dimensions, and ordered noncommuting propagation.
- Permit proven local optimizations and structural schedule selection while retaining generic dense affine fallback.

## Non-goals
No model/profile dispatch, whole-algorithm pattern, dedicated GDN execution, eager/token-serial fallback, or legacy engine routing.

## Dependencies
Issues 01–02.

## Acceptance
`StatefulOperator` runs generic and model-independent pressure programs without any profile-specific branch. Observable outputs match independent references for all four `{64,128} × {64,128}` Feature pairs. Generated execution uses parallel summary construction/composition; optimization misses do not reject valid supported programs.
