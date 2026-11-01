# 19: Replace both prefix scans with one streaming schedule

**What to build:** Compose per-token affine summaries without the two host-driven prefix implementations, so dense (rank-one) and diagonal (feature-wise decay) States both run in one launch whose cost is bounded by memory traffic.

**Blocked by:** 12/Add Rank One Factorized Plan; 17/Compose Optimized Forward and Backward.

**Status:** in progress (19a, 19b-i, 19b-ii and the chunked diagonal schedule are delivered; the 2x latency target, the packed form, and the rank-one chunk form remain)

Baseline (`testing/benchmark_stateful_plans.py`, H20, BF16, B1 H8 K=V=64):

| family | plan | T=2048 forward before | after 19a | forward+backward before | after 19a |
| --- | --- | --- | --- | --- | --- |
| `gla` | elementwise | 4.59 ms | 1.87 ms | 16.5 ms | 7.9 ms |
| `mamba2` | elementwise | 4.42 ms | 1.92 ms | 16.7 ms | 8.0 ms |
| `simple-gla` | scalar-factorized | 0.30 ms | 0.30 ms | 15.2 ms | 7.3 ms |
| `linear` | scalar-factorized | 0.30 ms | 0.30 ms | 15.5 ms | 7.6 ms |
| `gdn` | rank-one | 81.6 ms | 81.6 ms | 169.5 ms | 169.5 ms |
| `rwkv7` | rank-one | 81.6 ms | 81.6 ms | — | — |

19b-ii (delivered): the factored family now compiles with a backward. The forward records the per-token State (bf16) when a backward is present; one reverse-walk kernel reads those States, replays each token's authored chain forward to recover an operation's input State, and writes every gradient -- op factors and coefficients, injection factors, the readout operand, and the initial State. `_factored_vjp` and the `factored` backward context carry the stacked factors between the kernels, and the plan charges the recorded States as workspace only for backward-enabled compilations, so inference still runs with zero sequence-scaled workspace.

| family | T=2048 forward | T=2048 forward+backward |
| --- | --- | --- |
| `gdn` | 81.6 -> 4.65 ms | 169.4 -> 24.0 ms |
| `rwkv7` | 81.6 -> 5.01 ms | 169.5 -> 23.8 ms |

Evidence: `test_gdn_factored_recurrence_gradients_match_independent_reference` compares every gradient against `torch.autograd.grad` of the FP64 oracle at T=130; the residual is about 0.5% and comes from the bf16 State recording. The remaining backward throughput, the `19a`-family readout fusion, and the packed mode are the open items above.

19b-i (delivered): a `factored-recurrence` Plan Representation and `tilelang-factored-recurrence` lowerer add `_stateful_factored.py`, one kernel that keeps every State column in registers, replays the authored whole-State or first-feature Axis Scale and first-feature Rank One operations per token, adds the injection outer products, and contracts the readout. Eligibility (`rank_one_factored_eligible`) requires one Transition, one contracting first-feature readout, disjoint single-feature injection factors, and at least one rank-one contraction, so the diagonal-only Programs keep the scalar and elementwise kernels that already fuse the same work into fewer, shorter kernels. The family is forward-only: with a compiled backward the plan stays on the previous path, exactly as `scalar-factorized` does.

| family | T=2048 forward before | after 19b-i | launches | workspace |
| --- | --- | --- | --- | --- |
| `gdn` | 81.6 ms | 3.73 ms | 1 + 20 | 0 (was 384 KiB/token) |
| `rwkv7` | 81.6 ms | 4.00 ms | 1 + 18 | 0 |
| `gdn` T=8192 | rejected | 17.7 ms | 1 + 19 | 0 |

Evidence: `test_gdn_factored_recurrence_matches_independent_reference` and `test_gdn_factored_recurrence_continues_from_a_returned_state` compare against `gated_delta_rule_reference` at T=130 with an initial State and a prefix/suffix continuation.

19a (delivered): `_compile_elementwise_prefix` is now a streaming kernel. Every thread owns one State element and walks its segment accumulating `(A_t, B_t)` with `S_t = A_t S_0 + B_t`, so the cross-thread scan, its barriers, and the shared-memory chunk loop are gone; per-axis, per-token-scalar, and full elementwise factors all lower to the same accumulation. The kernel also composes in reverse (`reverse=True`), which removed the seven flipped token-sized copies the elementwise VJP used to make.

Measured effect of 19a: the three backward scans fall from 7.5-8.5 ms to 1.0 ms, forward falls 2.4x, and the whole training step falls ~2x. Against Flash Linear Attention at T=2048 (`testing/benchmark_linear_attention.py`, both sides VALID against the FP32 oracle) `gla` moves 250x -> 11.9x -> 5.84x end-to-end and 366x -> 16.4x on graph replay across 20 and 19a. The remaining 5.84x is the per-token State materialization and the token-serial accumulation, which 21 and 19b remove.

What remains for 19b, in the measured order of impact:

1. **Rank-one backward throughput**: the reverse walk is now the slower half (19 ms of the 24 ms training step at T=2048) because one block per sequence and head walks the whole sequence with 128 threads. Splitting the column axis into blocks, or the WY form below, is the next step. Packed mode also still falls back to the doubling scan.

   Two findings from 19b-i that shape 19b-ii: every per-token parameter gradient (`dL/dr`, `dL/dq`, `dL/dc`, `dL/d(scale)`) is a reduction across State columns, so the backward has to be tile based rather than thread-per-column like the forward; and the State must come from storage or a checkpointed replay, not from inverting the forward ops.

   A rejected design, recorded because it cost a full implementation: reconstructing `s_{t-1}` from `s_t` by inverting each operation (scales divide, injections subtract, a rank-one update inverts through Sherman-Morrison) needs no extra storage, but it is numerically unusable. A decaying State shrinks like `exp(-sum |g|)`, so the values the walk recovers are the *forward* States, which round to zero in FP32 well before the sequence ends; dividing the surviving low bits back up amplifies the rounding error instead of the signal. Measured against the FP64 oracle on the GDN profile: 1e-2 relative error at T=8 and 1e13 at T=130, with the kernel's arithmetic otherwise matching. 19b-ii must therefore store State checkpoints and replay the intra-token operation chain forward for the intermediates, or carry the WY matrix described below so intra-chunk prefixes have a closed form.

   Design note for the implementation: within a chunk, a chain of rank-one updates `prod_s (I + c_s l_s r_s^T)` is `I + A W B^T` with `A = [l_1..l_n]`, `B = [r_1..r_n]` and an `n x n` coefficient matrix `W` from a triangular solve (the WY represention). That makes the chunk summary `O(n^2 R)` instead of `O(n R^2)`, the chunk scan cheap (`T/n` summaries), and the per-token epilogue a prefix of the same form. The same factors serve the VJP: the state adjoint recurrences in reverse over `(l, r, c)` plus the per-token gradients `dL/dl = ..., dL/dr = ..., dL/dc = l^T S r`, none of which need a `[T,H,R,R]` tensor.
2. **Readout fusion (see 21)**: the elementwise family still materializes per-token States and then reads them back, which is why the VJP spends ~2 ms in token-sized torch passes and why GLA forward is 6x the scalar-factorized schedule. Fusing the contraction into the streaming kernel removes those passes entirely.
3. **Chunk-level parallelism**: the streaming kernel parallelizes over State elements only (heads x R*C/threads blocks), which is enough at K=V=64..128 but not at small State sizes.

- [x] The elementwise prefix composes in one launch with no cross-thread scan, for per-axis and scalar factors, in dense and packed modes.
- [ ] One chunked path serves dense and diagonal summaries with chunk-level parallelism.
- [x] The doubling scan is deleted or retained only as a CPU reference; no CUDA path reaches it. (`_segmented_affine_scan` and `_segmented_elementwise_scan` are called only from the `else` branches of the two `runtime.device.type == "cuda"` tests in `_lower_parallel_scan`; every CUDA path goes through `executable.prefix`.)
- [x] Authored propagation order, segmentation, `initial_state`/final-State semantics, packed mode, and gradient parity are unchanged against the current implementation, checked by the existing H20 suites. (`test_gdn_h20.py`, `test_fla_coverage.py`, `test_linear_attention_examples.py`, and `test_two_axis_elementwise.py`: 25 pass, including the packed and gradient cases.)
- [ ] `gdn`, `rwkv7`, `gla`, and `mamba2` land within 2x of the scalar-factorized group at T=2048 with one launch per pass.

**Chunked path attempt (reverted, blocker diagnosed).** The chunked schedule was implemented and is the right shape for this ticket; it was reverted before commit because one build could not be verified, and the blocker is precise enough to pick up directly.

Built and compiled: a boundary-carry pass (the existing elementwise prefix kernel compiled with a chunk stride, writing the running pair into a compact `[chunks, heads, rows, columns]` buffer instead of one row per token) plus `_compile_chunked_elementwise`, one fused kernel whose grid is `(sequences * heads, chunks, column blocks)`, whose State column lives in registers, and which replays the chunk's per-token decay, the injection outer product from its two factors, and the readout contraction, writing only the output. With `chunk_tokens=64` at T=2048, H=8, K=V=64 that is 256 blocks of 64 threads instead of the 8 blocks the factored kernel runs, and no per-token State touches global memory.

Blocker: compiling the same `kernel` symbol in two output layouts (per-token rows for the ordinary path, one row per chunk boundary for the carry) collided in TileLang's kernel cache. The compact call received the per-token build: `kernel kernel input output_bias shape[0] violates packed ABI constraint; expected: 32, got: 2048`. Fix options, cheapest first: give the boundary variant its own `@tilelang.jit` symbol (and its own `lru_cache` entry) so the two layouts never share a cache key, or avoid two layouts entirely by having the boundary pass write its rows into a separate output tensor. Either way the parity check to run first is the existing per-token path against the chunked entry on one chunk-aligned dense case.

The dispatch that was evaluated and removed with it: `_lower_parallel_scan` routes `axis-scale` forward-only compilations (CUDA, dense, uniform chunk-aligned length, single injection and readout, single per-axis factor) to the chunked entry, with `_token_major` as the declared transposition seam for per-token factors. The two-axis pair needs both compact carries, so it stayed on the per-token path in that attempt.

**Chunked path delivered (`_stateful_chunked.py`).** A diagonal Program on a chunk-aligned dense sequence now runs through two kernels that own their symbols and layouts, so they never share a TileLang cache entry (the collision that blocked the first attempt):

- `boundary_pair_kernel` walks each sequence once and writes the running `(A, B)` pair only where a chunk ends, into a compact `[chunks, heads, rows, columns]` buffer.
- `chunk_recurrence_kernel` gives every `(sequence, head, chunk, column block)` its own block, reads the pair from the previous boundary as `A * S_0 + B`, holds the State column in registers, replays the chunk's factors and the injection outer product from the two per-token factors, contracts the readout in place, and lets the last chunk write the final State.

No per-token State reaches global memory except the output, and the block count scales with the chunk count (at T=2048, H=8, K=V=64: 256 blocks of 64 threads against the factored kernel's 8).

Measured on an idle H20 (card 4, 4 MiB in use, 0% utilization), T=2048, H=8, K=V=64, inference:

| Program | before | chunked | scalar-factorized group |
| --- | --- | --- | --- |
| `gla` (row decay, T=2048) | 1.870 ms | 1.232 ms | 0.304 ms |

So `gla` moves from 6.2x to 4.05x of the scalar group: better, and not yet the 2x this ticket asks for. The remaining gap is the token-serial walk inside each chunk; the next lever is splitting the row axis across threads instead of holding the whole column in one thread. `gdn` and `rwkv7` are unchanged: they carry a rank-one propagation, which needs the WY form rather than this diagonal schedule.

Correctness: `tests/functional/test_chunked_elementwise.py` covers a row-decay Program at a chunk-aligned length (chunk-boundary tokens 63/64/127 and the final State), a two-axis Program through the same carry, and the scalar family's unchanged selection. Final-State error against the FP32 recurrence is 4.8e-6.

**Latency progress at T=2048, measured, band = 0.304-0.312 ms (`linear` 0.304, `simple-gla` 0.312, `retnet` 0.315):**

| family | start of this work | now | vs band | kernel time |
| --- | --- | --- | --- | --- |
| `gla` | 1.870 | **0.567 ms** | 1.9x | 0.232 ms |
| `mamba2` | - | **0.608 ms** | 2.0x | 0.242 ms |
| `gdn` | 4.568 | 4.568 ms | 15x | - |
| `rwkv7` | 4.987 | 4.987 ms | 16x | - |
| `rebased` (2 States, rank-1 denominator) | rejected | 3.306 ms | - | ~2.7 ms |

`gla`'s kernel time is now 0.232 ms against FLA's 0.236 ms on the same shape, so the remaining 0.335 ms is prologue: the per-token decay factors are computed in torch (an `exp` over `[1,8,2048,64]` plus its fp32 cast), which fusion into the kernel would remove. Three changes got there: the chunk-parallel boundary walk, keeping the per-axis factors separate instead of their outer product (268 MB for a 64x64 State at T=2048, 1 GiB for `rebased`'s 256x64), and trying the chunked schedule before assembling an injection the chunked path never reads.

`gdn` and `rwkv7` are untouched because their rank-one Transitions go through `factored_recurrence_dense`, which replays an interpreted operation list per token while holding the State column in registers: about 500x off FLOP-bound, so it needs the chunked WY form (chunk transition `I + L W R^T` via a triangular substitution, intra-chunk masked attention, matmul state update) rather than tuning.
