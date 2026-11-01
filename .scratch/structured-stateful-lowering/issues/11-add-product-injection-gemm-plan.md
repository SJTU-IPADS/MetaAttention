# 11: Add the Product Injection GEMM plan

**What to build:** Recognize role-compatible Product Injections as a reusable GEMM-oriented Mathematical Plan and compile equivalent forward and backward execution without algorithm-name assumptions.

**Blocked by:** 06/Migrate existing structured forward paths; 09/Enforce named forward-backward context contracts.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] Matching uses typed factor roles and expression structure rather than query, key, value, model, or profile names.
- [ ] Disjoint-role outer products, full-State direct factors, multiple injections, and non-square State Feature dimensions are handled deliberately.
- [ ] Shared Feature-role multiplication is recognized only when it is explicit in the Input Expression.
- [ ] The GEMM-oriented plan coexists with the generic dense affine plan and silently declines unsupported structures.
- [ ] Specialized outputs, final State, every input gradient, and initial-State gradient match generic and independent references.
- [ ] Adding or removing the candidate does not alter validation, Scanability, or public error categories.
