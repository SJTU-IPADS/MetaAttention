# 03: Complete the metadata-only compile cutover

**What to build:** Make metadata-only compilation the only Stateful Operator compile contract and migrate every repository caller while keeping tests, examples, benchmarks, and API documentation coherent.

**Blocked by:** 01/Establish metadata-only dense compilation; 02/Establish metadata-only packed compilation.

**Status:** delivered in code (audited 2026-09-26: every artifact the ticket names is present and exercised by the unit and functional suites; the ticket's own H20 acceptance evidence is not re-gathered in this pass)

- [ ] The public compile contract accepts role sizes, explicit Target, dense or packed mode, and independent optimized, autotune, and backward choices.
- [ ] Compile-time Tensor and offsets arguments are rejected; no compatibility alias or deprecated path remains.
- [ ] Old duplicate tuning options, including a separate backward-tuning choice, are removed in favor of the single autotune choice.
- [ ] Invoking an uncompiled StatefulOperator does not lazily compile or infer a specialization from runtime Tensors.
- [ ] Every test, example, benchmark, and public API description uses the metadata-only contract.
- [ ] Dense and packed public behavior remains green after the old form is removed.
