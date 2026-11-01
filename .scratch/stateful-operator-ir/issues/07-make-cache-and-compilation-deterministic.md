# Make Cache and Compilation Deterministic

Labels: `ready-for-agent`

## Goal
Define deterministic specialization, artifact identity, compilation, and publication for the compiler.

## Scope
- Include schema, canonical forward/backward IR, tensor metadata, dense/packed mode, active differentiable bindings, final-State mode, Target, and compilation options in semantic identity.
- Exclude runtime offset values, Python identity, source locations, messages, and device index.
- Add ABI/toolchain fingerprints to binary identity and exact H20/environment fingerprints to tuning identity.
- Reject unsupported layout/resources before cache/codegen; publish atomically under process-safe locking.
- Never persist transient compile/resource failures as negative entries.

## Non-goals
No sharing tuning records by compute capability alone and no public internal analysis report.

## Dependencies
Issues 01–06.

## Acceptance
Equivalent public invocations reuse artifacts; relevant semantic changes do not. Packed offset contents reuse one specialization when metadata matches. Concurrent compilation cannot expose partial artifacts. Transient failures recover on a later invocation.
