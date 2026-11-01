# 15: Benchmark Top Three Lowered Candidates

**What to build:** When autotuning is enabled, benchmark the statically strongest three Lowered Candidates plus the generic baseline on a versioned compiler-owned workload policy and select the lowest-latency valid candidate.

**Blocked by:** 14/Select feasible Lowered Candidates.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] The measured set contains the static Top Three and the generic baseline, with duplicates removed.
- [ ] Each candidate receives the same fixed small parameter-probe budget before cross-plan comparison.
- [ ] Dense policy covers multiple Batch and Sequence scales; packed policy covers uniform, long-tail, and empty-containing distributions.
- [ ] Users cannot supply workload points, weights, candidate counts, or selection objectives.
- [ ] Candidate score is the equal-weight geometric mean of per-workload latency normalized by the best valid candidate for that workload.
- [ ] A candidate that fails any required workload is removed; remaining candidates continue.
- [ ] Compilation time is recorded separately and excluded from the score; resource-invalid candidates never enter latency ranking.
- [ ] Autotuning disabled reads matching successful measurements but creates no new benchmark records.
