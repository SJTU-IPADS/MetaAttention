# 18: Publish layered caches and H20 evidence

**What to build:** Persist successful plan, tuning, forward, prepare, backward, and autograd-composition artifacts under their true contracts, then freeze the first representative workload policy and acceptance evidence on an idle H20.

**Blocked by:** 07/Support dynamic workspace and resource validation; 17/Compose independently optimized forward and backward.

**Status:** delivered in code, acceptance evidence open (audited 2026-09-26: layered publication exists; `H20_EVIDENCE.status` is still `pending`, so the workload-policy freeze and the statistical gates the ticket asks for are not met)

- [ ] Semantic, Mathematical Plan, Lowered Candidate, KernelConfig, context, tuning, and binary identities are distinct and deterministic.
- [ ] Dynamic Batch, Sequence, packed counts, offsets shapes, and offsets values do not enter binary identity; fixed roles, dtype, mode, Target, plan, lowerer, config, and context behavior enter the appropriate identities.
- [ ] Workload-policy version, exact H20 product, software stack, resource fingerprint, and clock policy isolate tuning records.
- [ ] Forward-only and backward-enabled compilation share a forward binary only when context production adds no different kernel behavior or cost.
- [ ] Forward, prepare, backward, and autograd-composition artifacts use process-safe locks and atomic publication; failures leave no partial or negative entries.
- [ ] H20 experiments freeze dense and packed workload-policy v1 and show the small probe budget does not systematically discard the final fastest plan.
- [ ] H20 evidence confirms dynamic runtime workloads do not trigger code generation, plan dispatch, or configuration dispatch.
- [ ] Existing statistical latency and peak-memory gates, the compositional pressure profile, and the full functional GPU suite pass.
- [ ] The specification and compositional-lowering ADR are updated with measured decisions and status only after evidence exists.
