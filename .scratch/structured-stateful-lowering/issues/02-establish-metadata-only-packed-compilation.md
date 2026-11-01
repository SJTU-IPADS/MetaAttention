# 02: Establish metadata-only packed compilation

**What to build:** Extend metadata-only Stateful Operator compilation to packed mode so one compiled callable supports dynamic logical-sequence counts, total-token counts, offsets shapes, offsets values, and valid empty sequences.

**Blocked by:** 01/Establish metadata-only dense compilation.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] Packed compilation requires fixed Head and Feature role sizes, explicit Target, and packed mode but no real input Tensor or offsets Tensor.
- [ ] Runtime validates token-major input contracts and offsets rank, int32 dtype, device, contiguity, start, end, monotonicity, and total-token agreement.
- [ ] One compiled callable executes correctly across different logical-sequence counts, total-token counts, and offsets shapes without replanning or recompiling.
- [ ] Uniform, long-tail, leading/interior/trailing empty, and all-empty packed invocations preserve sequence isolation and State ordering.
- [ ] Initial and final State sequence dimensions are validated against the runtime offsets length.
- [ ] Dynamic packed workload metadata does not change the compiled artifact identity.
