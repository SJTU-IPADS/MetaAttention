# 08: Generate generic compiler-derived backward

**What to build:** When backward generation is requested with structural optimization disabled, lower canonical compiler-owned Backward IR into a generic backward kernel and invoke it through ordinary PyTorch autograd.

**Blocked by:** 05/Execute generic dense forward through a Mathematical Plan.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] PyTorch autograd transports saved Tensors and cotangents but does not derive the backward algorithm or provide an eager fallback.
- [ ] Output-only, final-State-only, and joint cotangent paths produce gradients matching independent references.
- [ ] Missing output or final-State cotangents have mathematical zero semantics.
- [ ] Gradients are correct for every differentiable input, grouped Head Mapping reductions, and an explicitly provided initial State.
- [ ] Binding the same Tensor object to multiple input names accumulates all semantic gradient contributions.
- [ ] Compilation with backward disabled generates no backward kernel, and grad-enabled invocation fails explicitly.
- [ ] Dense, packed, empty-sequence, and inactive-tail backward behavior is verified through the public autograd seam.
