# 01: Establish metadata-only dense compilation

**What to build:** Let users compile a dense Stateful Operator from fixed Head and Feature role sizes, Target, and dense mode without supplying real Tensors. The returned callable must run the same compiled implementation across dynamic Batch sizes and dense Sequence lengths while preserving the existing ExecutionResult contract.

**Blocked by:** None (can start immediately).

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] Compilation accepts fixed sizes only for declared HeadRole and FeatureRole values and rejects Batch, Sequence, missing, unknown, or inconsistent role sizes.
- [ ] Algorithm IR remains the sole source for input names, logical roles, dtypes, State shape semantics, readout dtypes, and Head Mapping.
- [ ] Runtime invocation validates fixed role sizes, dtype, explicit Target, canonical contiguous layout, and dense input-role consistency.
- [ ] One compiled callable executes correctly for at least two Batch sizes and multiple positive Sequence lengths without planning or compiling again.
- [ ] Outputs, optional final State, and structured errors are verified through the public compile-and-invoke seam.
- [ ] The existing tensor-based compile entry remains temporarily usable through the same internal compile contract so the repository stays green during expansion.
