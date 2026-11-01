# 06: Migrate existing structured forward paths

**What to build:** Express the existing elementwise and scalar-factorized forward implementations as explicit Mathematical Plans and registered Lowered Candidates instead of implicit analysis flags or central dispatch branches.

**Blocked by:** 05/Execute generic dense forward through a Mathematical Plan.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] Elementwise affine structure produces an explicit Mathematical Plan and applicable lowerer.
- [ ] Scalar-factorized structure produces an explicit Mathematical Plan and applicable lowerer.
- [ ] The generic dense affine candidate remains available whenever either structured candidate is present.
- [ ] Candidate mismatch or unsupported concrete specialization selects another valid candidate rather than rejecting the Scanable Program.
- [ ] Lowerers implement the supplied Mathematical Plan without introducing undeclared algebraic rewrites.
- [ ] Existing acceptance profiles preserve forward outputs, final State, dense behavior, and packed behavior through the public seam.
