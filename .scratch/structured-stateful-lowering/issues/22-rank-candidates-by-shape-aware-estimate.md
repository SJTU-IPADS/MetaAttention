# 22: Rank candidates by a shape-aware estimate

**What to build:** Rank Lowered Candidates by their estimated cost for the concrete Feature sizes and workload instead of by the hard-coded per-lowerer constants, so a Program whose summary is diagonal cannot select a dense-scan candidate.

**Blocked by:** 14/Select feasible lowered candidates; 19/Replace the dense doubling prefix scan.

**Status:** complete

Evidence: `select_candidate` ordered by `LOWERERS[...].static_cost` (`tilelang-elementwise` 30 beats `tilelang-generic` 100 and `reference-stateful` 200 regardless of shape), while `CandidateEstimate` computed only a workspace number used for feasibility filtering.

Delivered: `CandidateEstimate` now carries the three components the ranking needs -- `bytes_per_token`, `flops_per_token`, and `launches` -- and `cost` is their weighted sum (`_FLOPS_PER_BYTE`, `_BYTES_PER_LAUNCH`). `CandidateLowerer.estimate` prices each Plan Representation from its own execution shape and the concrete Feature sizes: the summary families move their summary, the State they write, and the output over three passes; the factorized families move only their factors over one pass; the dense and reference families also move the summary matrices and the union layout. The workspace component of the estimate is now the single source of truth for both feasibility and the published `WorkspacePolicy`, so a plan can no longer compile under a charge its runtime does not honour.

- [x] The estimate covers bytes moved per token, per-token arithmetic, and expected launches, derived only from typed IR structure and concrete Feature sizes.
- [x] Ranking uses the estimate; the deterministic lowerer/identity tie-break stays.
- [x] `optimized=False` still selects generic execution only, and an infeasible candidate still leaves the Program valid with a lower-ranked plan.
- [x] A test proves that a Program with a feature-wise decay or a rank-one propagation never selects a dense-doubling candidate.
- [x] Selection decisions and their estimates are inspectable from the compiled operator (`TileLangPlan.candidate_estimates`, ordered by cost).
- [x] The scalar-factorized and rank-one families keep their current selections and measured latency.

Measured selections after the change (H20, BF16, B1 H8 K=V=64, T=2048, from `testing/benchmark_linear_attention.py` and the training-timing probe): `linear` and `simple-gla` keep `tilelang-scalar-factorized` (cost 16416, next candidate 411648), `gla` keeps `tilelang-elementwise`, `gdn` and `rwkv7` keep `tilelang-factored-recurrence` (cost 49152 against 2662400 for the elementwise summary). Latency is unchanged against the recorded baseline: 0.302/0.315/1.880/1.931 ms forward for `linear`/`simple-gla`/`gla`/`mamba2`, 4.645/5.038 ms for `gdn`/`rwkv7`, and 23.7 ms training for `gdn`.

Two consequences worth recording. A per-axis decay now has no zero-workspace candidate, so a one-byte envelope fails compilation with `COMPILE_RESOURCE` instead of silently selecting a plan that cannot run. And because the workspace component is monotone in the estimate, a "cheapest candidate is infeasible" case cannot arise for the v1 families; the mechanism is tested directly through `CandidateLowerer.lower`, and the reference lowering remains the family where an infeasible-only candidate is reachable.

Tests: `tests/unit/test_stateful_selection.py` (six cases: diagonal, rank-one, scalar, inspectability, infeasible candidate, `optimized=False`), `tests/unit/test_stateful_workspace.py`, and the migrated selector cases in `tests/unit/test_stateful_operator.py`.
