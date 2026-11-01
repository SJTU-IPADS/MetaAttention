# 16: Autotune the winning KernelConfig

**What to build:** After one Lowered Candidate wins plan comparison, perform full TileLang configuration search only within that candidate and publish one fixed KernelConfig for runtime execution.

**Blocked by:** 15/Benchmark Top Three Lowered Candidates.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] KernelConfig remains separate from Mathematical Plan and Lowered Candidate identity.
- [ ] Full search runs only for the selected candidate; losing candidates receive no full search budget.
- [ ] Forward and requested backward implementations may use independently tuned configurations.
- [ ] Autotuning disabled uses a matching successful cached configuration or the lowerer's stable default without new measurements.
- [ ] Successful configuration and measurements are persisted under Target, environment, candidate, and workload-policy identity.
- [ ] Failed configuration attempts do not create persistent negative cache entries or partial artifacts.
- [ ] Invocation uses the fixed selected configuration without runtime retuning or configuration dispatch.
