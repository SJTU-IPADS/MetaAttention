# 17: Compose independently optimized forward and backward

**What to build:** Combine Axis Scale, Product Injection/GEMM, Rank-One factorization, and shared State Contraction plans, then publish one autograd-compatible operator whose independently selected forward and backward implementations have a valid named context contract.

**Blocked by:** 11/Add the Product Injection GEMM plan; 12/Add the Rank-One factorized plan; 13/Add the shared State Contraction plan; 16/Autotune the winning KernelConfig.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] The planner composes legal non-conflicting representation rewrites and excludes mutually incompatible combinations.
- [ ] Forward and backward may select different Mathematical Plans, lowerers, and KernelConfigs while remaining equivalent to canonical forward and Backward IR.
- [ ] Only context-compatible forward/backward pairs remain eligible; missing values may be explicitly recomputed.
- [ ] A fixed prepare kernel may materialize named context when that yields a valid faster composition; it is never a runtime fallback.
- [ ] Structured backward failure can select another structured or generic backward without invalidating a valid forward.
- [ ] The compositional pressure profile compiles without whole-program, model, or profile matching.
- [ ] Dense and packed outputs, State, cotangents, input gradients, and initial-State gradients match independent references.
