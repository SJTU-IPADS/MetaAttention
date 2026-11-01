# 13: Add the shared State Contraction plan

**What to build:** Recognize independent State Contractions that observe the same post-transition State and enumerate a shared-readout Mathematical Plan with equivalent named outputs and gradients.

**Blocked by:** 06/Migrate existing structured forward paths; 09/Enforce named forward-backward context contracts.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] Matching preserves each readout name, declared order, contracted Feature role, retained Feature role, and output dtype.
- [ ] Fusion introduces no dependency between readouts and never changes their post-transition State semantics.
- [ ] A single readout or incompatible readout group does not produce an unnecessary fusion plan.
- [ ] The fused plan coexists with generic dense execution and declares its rewrite conflicts.
- [ ] Every fused output and corresponding gradient matches generic and independent references.
- [ ] Readout fusion matching uses typed structure rather than model, profile, frontend, or input names.
