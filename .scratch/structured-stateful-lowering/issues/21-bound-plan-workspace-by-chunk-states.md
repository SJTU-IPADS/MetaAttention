# 21: Bound plan workspace by chunk States, not by tokens

**What to build:** Every specialized plan holds only chunk-level States and the output, so workspace stops growing with the token count past the chunk boundary.

**Blocked by:** 19/Replace the dense doubling prefix scan.

**Status:** complete

Evidence: every plan reports the same 384 KiB/token (`WorkspacePolicy.bytes_per_token`), i.e. the per-token State `[T,H,K,V]` plus its scale/bias copies are materialized for the whole sequence. At T=8192 that is ~3 GiB, so 10 of 11 families are rejected with `RUNTIME_RESOURCE` before running; `linear` at T=2048 reads and writes 268 MB per pass (0.21 ms ≈ HBM roofline) for a 4 MB output.

- [x] The `tilelang-factored-recurrence` family keeps only the output and the final State: `bytes_per_token == 0` in inference, so it is accepted at T=32768. A compiled backward records the per-token State it replays from and charges `state_heads * rows * columns * 2` per token, which puts the 1 GiB envelope at T≈4096 for 8 heads of 128x128.
- [x] A regression test (`tests/unit/test_stateful_workspace.py`) pins the per-token charge of the factored and the summary plans and the token count where each stops fitting the envelope.
- [x] Chunk States, chunk summaries, and the output are the only sequence-scaled buffers for the summary families too; per-token summaries are consumed inside the kernel that produces them. (For a chunk-aligned dense workload: the chunked schedule keeps only the compact `[chunks, heads, rows, columns]` pair.)
- [ ] Every coverage family compiles and runs at T=32768 in dense and packed modes within the H20 envelope. (The diagonal dense families now do: a `gla` plan charges 2048 bytes per token at chunk-aligned lengths, so T=32768 needs 64 MiB and is accepted where it was rejected. Packed and the v2/v3 families are still governed by the per-token bound.)
- [ ] Output and final State are unchanged against the pre-change implementation for every family the functional suite covers.

**Measured gap (T=512, H=8, K=V=64, `testing/benchmark_linear_attention.py`).** Kernel-only graph replay separates the schedule from launch and readout overhead:

| Program | MetaAttention kernel | stock FLA kernel | MetaAttention end to end | stock FLA end to end |
| --- | --- | --- | --- | --- |
| `simple-gla` (scalar decay, scalar-factorized) | 0.062 ms | 0.024 ms | 0.135 ms | 0.253 ms |
| `gla` (per-feature decay, elementwise) | 0.450 ms | 0.044 ms | 0.690 ms | 0.330 ms |

Two separate gaps. The `gla` kernel is 10.2x off `chunk_gla`, which is the schedule difference the design note below explains, and its end-to-end time carries a further 0.24 ms because the elementwise family materializes the per-token State and contracts the readout in a separate torch pass -- exactly the buffer this ticket removes. The scalar-decay kernel is 2.6x off `chunk_simple_gla` but wins end to end, and stock FLA is reported INVALID against the oracle there (its toggle semantics differ), so only the `gla` row is a like-for-like baseline.

**Where the per-axis cost actually is (measured, T=512, H=8, K=V=64, idle H20).** The `gla` forward is three passes over one `[T, H, 64, 64]` fp32 tensor, each materializing it in global memory:

| stage | time |
| --- | --- |
| dense injection outer product (torch, builds `[T,H,R,C]`) | 94 us |
| elementwise prefix kernel (writes the per-token State) | 102 us |
| readout einsum (reads the per-token State) | 77 us |

No stage is pathological; each runs near its own traffic, and their sum is what one fused pass should replace. Two fused candidates measured against them:

| schedule | time | note |
| --- | --- | --- |
| `tilelang-factored-recurrence` kernel, same chain and shape | 588 us | one State column per thread in registers, `sequence_count * heads` = 8 blocks with 64 of 128 threads active |
| elementwise prefix kernel, same chain and shape | 102 us | thread per State element, `heads * cdiv(R*C, 128)` = 256 blocks |

The register-per-column shape is therefore *not* the schedule to copy as it stands: it gives up 64x of the parallel width (512 active lanes against 32768) and loses 5.8x for identical per-token work. Widening the factored eligibility to per-axis chains was implemented, measured at 0.690 ms -> 0.990 ms end to end, and reverted.

**Design note.** The lever is block count, not per-element arithmetic (~192 flops per thread per token). The chunked form is what buys blocks: take the per-chunk State carry from a cheap first pass over `[chunks, heads, R, C]`, then give each `(chunk, column block)` pair its own block with the State in registers and a thread-local readout row sum. A per-feature decay is compatible with the tensor-core form after all: inside a chunk, pre-scale the operands (`q' = diag(exp(P_t)) q`, `k' = diag(exp(-P_s)) k`) so that `q'_t . k'_s == q_t . diag(exp(P_t - P_s)) . k_s`, and the chunk-end summary becomes `diag(exp(-P_end)) * sum_s k'_s v_s^T`. That is the shape a per-axis chunked kernel should take.

**Chunked path attempt (reverted, blocker diagnosed).** The chunked schedule was implemented and is the right shape for this ticket; it was reverted before commit because one build could not be verified, and the blocker is precise enough to pick up directly.

Built and compiled: a boundary-carry pass (the existing elementwise prefix kernel compiled with a chunk stride, writing the running pair into a compact `[chunks, heads, rows, columns]` buffer instead of one row per token) plus `_compile_chunked_elementwise`, one fused kernel whose grid is `(sequences * heads, chunks, column blocks)`, whose State column lives in registers, and which replays the chunk's per-token decay, the injection outer product from its two factors, and the readout contraction, writing only the output. With `chunk_tokens=64` at T=2048, H=8, K=V=64 that is 256 blocks of 64 threads instead of the 8 blocks the factored kernel runs, and no per-token State touches global memory.

Blocker: compiling the same `kernel` symbol in two output layouts (per-token rows for the ordinary path, one row per chunk boundary for the carry) collided in TileLang's kernel cache. The compact call received the per-token build: `kernel kernel input output_bias shape[0] violates packed ABI constraint; expected: 32, got: 2048`. Fix options, cheapest first: give the boundary variant its own `@tilelang.jit` symbol (and its own `lru_cache` entry) so the two layouts never share a cache key, or avoid two layouts entirely by having the boundary pass write its rows into a separate output tensor. Either way the parity check to run first is the existing per-token path against the chunked entry on one chunk-aligned dense case.

The dispatch that was evaluated and removed with it: `_lower_parallel_scan` routes `axis-scale` forward-only compilations (CUDA, dense, uniform chunk-aligned length, single injection and readout, single per-axis factor) to the chunked entry, with `_token_major` as the declared transposition seam for per-token factors. The two-axis pair needs both compact carries, so it stayed on the per-token path in that attempt.

**Chunked path and the workspace bound.** The chunked kernels keep only chunk States: at T=2048, H=8, K=V=64 the live buffer is `chunks * heads * rows * columns * 4` = 4 MiB total, about 2 KiB per token, against the 384 KiB per token the published `WorkspacePolicy` still charges for the elementwise family. The kernels therefore meet this ticket's bound while the policy does not yet encode it. Next step: teach `CandidateLowerer.estimate` about the chunked path so the published policy charges `heads * rows * columns * 4 / chunk_tokens` per token when that path is reachable, which also needs the estimate to see the same gate the runtime uses (dense, chunk-aligned, single injection and readout, no gradient).

**Delivered: the policy encodes the chunk bound.** `WorkspacePolicy` now carries an optional `aligned_bytes_per_token` with its `alignment_tokens`, and `required_bytes` uses it exactly when the runtime workload divides the chunk length; every other workload is bounded by `bytes_per_token`, which covers the per-token fallback the same plan still takes for an unaligned sequence length. `CandidateLowerer.estimate` reports both numbers for the `AXIS_SCALE` representation when the compilation can reach the chunked schedule (dense, no compiled backward, diagonal summary, CUDA target, one Transition, one injection, one readout), and `compile_plan` publishes them from the selected estimate.

Measured for a `gla` plan at K=V=64 with 8 heads: 2048 bytes per token at chunk-aligned lengths against 393216 bytes per token for the per-token bound. T=2048 needs 4 MiB and T=32768 needs 64 MiB, both accepted, where the previous policy rejected T=32768 outright. A 2050-token workload still charges the per-token bound, so the declared envelope never under-covers a reachable path. `tests/unit/test_stateful_workspace.py` pins the aligned, aligned-at-scale, and unaligned cases.
