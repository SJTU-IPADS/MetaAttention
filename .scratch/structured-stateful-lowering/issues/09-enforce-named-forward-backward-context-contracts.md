# 09: Enforce named forward-backward context contracts

**What to build:** Connect generated generic forward and backward implementations through compiler-owned named context schemas and reject incompatible pairs during compilation rather than invocation.

**Blocked by:** 08/Generate generic compiler-derived backward.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] Context values use stable semantic paths and declare dtype, logical roles, and shape rules.
- [ ] Forward candidates declare produced context; backward candidates declare required and explicitly recomputable context.
- [ ] A pair is accepted only when every required value is produced or recomputable, including dtype, role, and shape compatibility.
- [ ] Context may contain inputs, outputs, token State, chunk State, affine summaries, and gates, including values whose storage grows with Sequence length.
- [ ] Context is stored only through the generated autograd wrapper and is absent from ExecutionResult and the public compile API.
- [ ] Resource-infeasible context removes the candidate pair before artifact publication.
- [ ] Public gradient tests cover saved, recomputed, missing, and incompatible context cases.
