# 20: Lower per-axis diagonal decay elementwise

**What to build:** Represent a single-role Axis Scale as a per-axis elementwise factor instead of a dense diagonal matrix, so a feature-wise decay never materializes `[T,H,K,K]`.

**Blocked by:** 10/Add Axis Scale Structured Plan.

**Status:** complete

Evidence (before): `_lower_parallel_scan` set `needs_dense_transform` whenever a propagation was a rank-one or a single-role `AxisScale`, and `_axis_scale_summary` then built `torch.diag_embed(factor) @ identity`, i.e. a `[T,H,K,K]` tensor per token. The elementwise prefix kernel already resolved `scale[token, head, scale_row, scale_column]` with `scale_row = 0 if scale_rows == 1 else row`, so per-axis factors were supported by the kernel but rejected by the summary builder and the Python shape guard.

Delivered:

- `_dense_summary_required` is the single predicate for the dense-summary decision, used by both `_compile_plan` (backward context family) and `_lower_parallel_scan` (forward kernel choice). A rank-one propagation or two differently scaled axes still needs the dense transform; one scaled axis does not.
- `_axis_scale_summary` lowers a first-feature scale to `[T,H,K,1]` and a second-feature scale to `[T,H,1,V]`, and rejects (rather than silently densifying) a composition of two different scaled axes.
- `tilelang_elementwise_prefix` accepts `(1,1)`, `(K,1)`, `(1,V)` and `(K,V)` scale shapes.
- Two kernel defects surfaced and are fixed: the write block used `scale_row`/`scale_column` outside their defining `if` region, and the chunk carry only refreshed `carry_scale` for `(1,1)` scales, so any per-axis or full-elementwise scale composed wrongly across a chunk boundary.
- `tests/functional/test_linear_attention_examples.py` gained `test_linear_attention_dsl_gradients_match_fp32_recurrence`, which covers the elementwise VJP for a scalar gate, a per-axis gate, and no gate; the per-axis scale gradient reduces `[T,H,K,1]` through `_sum_to_shape`.

Measured on H20 (BF16, B1 H8 K=V=64, `testing/benchmark_stateful_plans.py`):

| family | T=512 before | T=512 after | T=2048 before | T=2048 after | launches @2048 |
| --- | --- | --- | --- | --- | --- |
| `gla` | 16.5 ms | 1.34 ms | 80.0 ms | 4.59 ms | 44 -> 1 |
| `mamba2` | 16.5 ms | 1.37 ms | 80.0 ms | 4.42 ms | 44 -> 1 |

Against the library baseline at T=2048 (`testing/benchmark_linear_attention.py`, both sides VALID against the FP32 oracle): `gla` moves from 250x to 11.9x Flash Linear Attention end-to-end and from 366x to 41x on graph replay.

- [x] A first-feature Axis Scale produces `[T,H,K,1]` and a second-feature Axis Scale `[T,H,1,V]`; neither builds a diagonal matrix.
- [x] The elementwise shape guard admits per-axis factors and still rejects malformed ones.
- [x] `gla` and `mamba2` select the elementwise plan without a dense transform; `retention_attention` and `simple_gla_attention` keep their scalar path.
- [x] Forward outputs, final State, and gradients match the FP32 recurrence for scalar and per-axis gates.
- [ ] Two single-role Axis Scales compose as a row factor times a column factor without densifying: still dense, moved to 24.
- [ ] Workspace for axis-scale Programs is `O(T·H·(K+V))` plus the State output: the summary is now elementwise, but the plan still materializes the per-token State, so this is measured under 21.
- [ ] Within 2x of the scalar-factorized group at T=2048: still 15x, because the elementwise prefix kernel scans per token with four matrix lanes per block. The chunked schedule that closes this is 19.
