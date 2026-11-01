# 23: Bring v2/v3 vocabulary Programs onto the fast path

**What to build:** Give multi-State, rank-1, product-Feature, and expression-readout Programs the same chunk-summary execution the specialized v1 plans get, so the reference lowering stops being their only executor.

**Blocked by:** 19/Replace the dense doubling prefix scan; 21/Bound plan workspace by chunk States; 02/Establish metadata-only packed compilation.

**Status:** in progress (the per-State layout box is verified; per-axis scales elementwise, the streaming State scan, and the per-State charge are delivered; the multi-State chunk scan, the readout epilogue, and the remaining envelope case are open)

Baseline (`testing/benchmark_stateful_plans.py`, H20, BF16, B1 H8 K=V=64) against 0.14–0.32 ms for the specialized plans:

| family | T=512 | T=2048 | note |
| --- | --- | --- | --- |
| `normalized-linear` | 4.8 ms | rejected | `RUNTIME_RESOURCE` |
| `gsa` | 9.6 ms | 40.0 ms | 0/323 launches |
| `rwkv6` | 9.9 ms | 44.7 ms | 0/173 launches |
| `hgrn` | 2.2 ms | 2.8 ms | elementwise scan |
| `based`, `rebased` | rejected | rejected | `RUNTIME_RESOURCE` |

`feature_layout` builds one union layout over every Input and State Feature role and the reference lowering materializes per-token States in that layout, which is both the workspace cliff and the reason a product-Feature Program cannot run at all.

- [ ] Each State carries its own layout and shape; no union `[T,H,*roles]` tensor spans the Program.
- [ ] Rank-1 and per-axis diagonal States reuse the elementwise chunk scan; dense States reuse the chunk-summary scan.
- [ ] `StateTerm`, `Contract`, `Outer`, `Softmax`, and prior-State readouts are evaluated against chunk States in the epilogue, so the readout adds no full-sequence pass.
- [ ] Every v2/v3 coverage family compiles and runs at T=4096 within the H20 envelope, and each is at most 3x the specialized plan for the same State shape.
- [ ] CPU parity tests and the H20 coverage parity tests keep passing unchanged, and per-family H20 numbers are recorded as evidence.

**Implementation map (surveyed, not yet built).** The union layout has exactly two consumers, both in `_stateful_reference.py`:

- `feature_layout` (line 105) builds the ordered union of every Feature role over inputs and States and returns the axis map.
- line 155 lays out *inputs* as token-major `[tokens, heads, *union features]`; line 260 passes the same axis map into `_scan_state`, which is where the per-token State is materialized in that union shape.

So the first box is narrower than it looks: the input layout can stay union-based (inputs really are shared per Program), and what has to change is that `_scan_state` allocates and carries each State with its *own* `state.feature_roles` order and shape. The expression evaluator already keys everything by role (`axes[role]`), so the change is a substitution at the state-term sites -- `StateTerm`, `Contract`, `Outer`, `Softmax` -- from the union axis map to a per-State one, with both maps available where a State term and an Input term meet.

Payoff, in the order it unblocks the rest of this ticket: a State no longer pays for Feature roles it does not use, the per-token State buffer shrinks to `[tokens, heads, *own roles]`, and that per-State shape is exactly what the chunk scan needs to take over from the reference lowering.

Scope correction on this ticket's second box. The chunked schedule from ticket 19 reached the *v1* diagonal families only: both the compile-time eligibility and the runtime gate require one Transition, one injection, and one readout, so no multi-State or multi-readout Program can reach it. "Rank-1 and diagonal States on the elementwise chunk scan" is therefore untouched for the v2/v3 vocabulary -- `hgrn` is the one coverage family the elementwise family serves today, and only because it is single-Transition. This box follows the per-State layout work rather than preceding it. Then the readout epilogue can be evaluated against chunk States, and the `_stateful_chunked.py` schedule already built for ticket 19 extends to multi-State Programs once each State carries its own carry buffer.

**First box verified satisfied; the ticket's premise was stale.** Reading the code rather than trusting the note: `_scan_state` allocates each per-token State at `[tokens, heads, left_size, right_size]` taken from that State's own `view.feature_roles`, `_injection_terms` builds the per-token injection at the same sizes, and `_contract_state_term` contracts by `state_spec.feature_roles` (`thf,thfv->thv` for a first-feature contraction). The union `feature_axes` map is used only to locate a role inside the *input-derived operand*, which genuinely carries several roles. So "each State carries its own layout and shape; no union `[T,H,*roles]` tensor spans the Program" already holds; the union layout spans inputs, by design, not States.

That means the ticket's stated evidence ("the reference lowering materializes per-token States in that layout, which is ... the reason a product-Feature Program cannot run at all") is wrong about the mechanism. The real blockers, in order:

1. `_affine_summary` sets `dense = any(rank-one or single-role AxisScale over the propagations)` and then builds dense `[T,H,R,R]` and `[T,H,C,C]` factors for any State whose Transition carries a per-axis scale. That is exactly the problem tickets 20 and 24 removed for the v1 families, and it is why `hgrn`, `gsa` and `rwkv6` are slow. The v1 answer -- one elementwise factor per scaled axis, with the pair composed per element -- applies unchanged here, and this is the highest-value step in the ticket.
2. `based` and `rebased` are rejected at T=512 by the workspace envelope, not by a layout rule: a Product Feature State makes `dense = rows^2 + columns^2 + rows*columns` large in the estimate, and the reference branch charges `heads * (2 * dense + 4 * states) * 4` per token. Fixing (1) shrinks that charge directly for the diagonal cases; the product-feature cases then need the estimate priced against what the kernel actually materializes.
3. Per-token State materialization itself (the workspace cliff and the readout pass) is what the chunk scan removes, and that is the second half of this ticket, as the earlier note says.

So the first box closes by inspection; the work starts at `_affine_summary`.

**Delivered: per-axis scales stay elementwise in the reference lowering.** `_affine_summary` no longer forces dense `[T,H,R,R]`/`[T,H,C,C]` factors for a single-feature Axis Scale; it composes one factor per axis exactly as tickets 20 and 24 do for the v1 families, and `_scan_state` only asks for dense matrices when a rank-one contraction is present. The reference estimate was updated with it: without a rank-one contraction it now charges the factor, the injection, and the per-token State instead of the dense pair (previously it over-charged because the model assumed the matrices were still there, which briefly pushed gsa and rwkv6 past the envelope at T=2048).

Measured with `testing/benchmark_stateful_plans.py` on an idle H20 (card 4):

| family | T=512 before | T=512 after | T=2048 before | T=2048 after | workspace/token (T=2048) |
| --- | --- | --- | --- | --- | --- |
| `gsa` | 9.6 ms | 2.66 ms | 40.0 ms | 8.16 ms | 192 KiB |
| `rwkv6` | 9.9 ms | 3.61 ms | 44.7 ms | 14.80 ms | 384 KiB |
| `hgrn` | 2.2 ms | 1.24 ms | 2.8 ms | 1.45 ms | 0.8 KiB |
| `normalized-linear` | 4.8 ms | 4.40 ms | rejected | rejected | 768 KiB |
| `based` | rejected | 17.2 ms | rejected | 66.2 ms | 36 KiB |
| `rebased` | rejected | rejected | rejected | rejected | - |

`based` and `rebased` are the product-Feature cases: `based` now runs where it was rejected, `rebased` still exceeds the envelope. Verified by the H20 coverage suites (`test_fla_coverage.py`, `test_gdn_h20.py`, `test_linear_attention_examples.py`, `test_chunked_elementwise.py`: 25 pass) and 153 unit tests.

What is left in this ticket: the readout still costs one torch op per token (90-654 launches per family per the table above), so the readout epilogue against chunk States is the next lever, and with it the chunk scan for multi-State Programs. `rebased` needs the product-Feature charge revisited once its readout stops dominating.

**Delivered: the reference State scan is the streaming kernel.** `_scan_state` no longer runs the host-driven doubling scan for an elementwise summary on CUDA. It calls the same streaming prefix kernel the v1 elementwise family uses, gated on no bound value requiring grad so a differentiable call keeps the torch scan and autograd is untouched. That removes the per-halving passes and their token-sized intermediates, which is where the launch counts in the table below came from.

Measured with `testing/benchmark_stateful_plans.py --seqlen 2048` on an idle H20 (card 4), cumulative across this ticket's two commits:

| family | baseline | after elementwise summary | after streaming scan | launches (baseline -> now) |
| --- | --- | --- | --- | --- |
| `gsa` | 40.0 ms | 8.16 ms | **2.13 ms** | 220 -> 48 |
| `rwkv6` | 44.7 ms | 14.80 ms | **2.81 ms** | 122 -> 36 |
| `hgrn` | 2.8 ms | 1.45 ms | **0.58 ms** | 106 -> 20 |
| `based` | rejected | 66.2 ms | **11.36 ms** | 654 -> 139 |
| `normalized-linear` | rejected | rejected | rejected | - |

Against 0.14-0.32 ms for the specialized plans, that leaves `hgrn` inside this ticket's 3x bar and `gsa` (6.7x) and `rwkv6` (8.8x) still above it, where they started at 125x and 140x. `normalized-linear` is now the outlier: it still exceeds the envelope at T=2048 even though its siblings with the same shape do not, so its per-State charge needs the same treatment the others just got. The remaining structural levers are unchanged: the readout is still evaluated over the materialized per-token State rather than against chunk States, and multi-State Programs cannot reach the chunked schedule.

**Delivered: the reference charge is per State.** The estimate priced the reference lowering as `first_state_size * state_count`, so a Program with a rank-2 numerator and a rank-1 denominator paid twice for the larger one and `normalized-linear` was rejected at T=2048 while its siblings with the same structure ran. `EstimationShape` now carries the sum of each State's own element count and the reference branch charges that. Measured at T=2048 on an idle H20: `normalized-linear` compiles and runs in 3.02 ms (was `RUNTIME_RESOURCE`), with `gsa` 2.10, `rwkv6` 2.83 and `hgrn` 0.63 ms unchanged.

All five non-product coverage families now run at T=2048 (`rebased` was still rejected in the last full sweep and is the remaining envelope case). Against the 0.14-0.32 ms specialized band that is `hgrn` 2.0x, `gsa` 6.6x, `rwkv6` 8.8x and `normalized-linear` 9.4x, so three families are still above this ticket's 3x bar. Their remaining cost is the readout, which is still evaluated over the materialized per-token State rather than against chunk States -- the next item.

**`rebased` rejection is honest, not a mis-charge.** Measured at T=512 it runs in 2.87 ms but declares 1560 KiB/token, so T=2048 is 3.2 GB and correctly exceeds the 1 GiB envelope. That number is the per-State sum the reference lowering genuinely materializes (elementwise factor, injection, and per-token State for each of its States), not the `first_state * state_count` artefact `normalized-linear` had. So there is nothing to re-price here: `rebased` becomes reachable at T=2048 exactly when multi-State Programs reach the chunked schedule, whose chunk States cost `rows * columns / 64` elements per token -- about 1 KiB/token for this shape, against 1560.

That makes the three remaining boxes one piece of work: extend `_stateful_chunked.py` from one State to many, give each State its own carry pair, and evaluate the readout epilogue for contractions against chunk States (expression readouts over several States need the per-token State and would keep the reference path until a chunk-level form exists for them).

**Note for pinning the estimate per family.** A first attempt at a regression test that pins each family's declared charge was reverted mid-edit rather than committed half-applied, but it produced one useful datum: `hgrn` at `heads=1, dim_value=64` declares 6144 bytes per token, which is `1 * 3 * 512 * 4` -- so its State carries 512 elements, not the 64 a single per-Feature vector would suggest (its roles include the input feature axis alongside the state axis). A per-family pin must therefore read the roles from the operator and derive the expected charge from them, rather than assuming one element per role; the two estimate bugs this session were both silent capability changes, so such a pin is worth having.

**Corrected scope (read from the code after the checklist below was written; the checklist's central assumption is wrong).** The elementwise/chunked family is single-State *by construction*, so there is no per-State loop to widen. The enclosing method that calls `_lower_chunked_elementwise` returns `(output,)` and builds `StateTuple((state_spec.name,), ...)` for one `state_spec`, and the dispatch that reaches it requires `family == "axis-scale"`. Multi-State Programs such as GSA and normalized-linear never enter it: they reach the reference lowering, which is why they measure 2.10 ms and 3.02 ms today rather than being chunked. `_lower_chunked_elementwise` is therefore not on the path for any of these families, and widening its gate would change nothing.

**What multi-State chunking actually needs.** `examples/gated_slot_attention.py` is the clearest case. It declares two States, `key_slot [K, slot]` and `slot_value [slot, V]`, both decayed by the same per-slot gate (which therefore appears as a column factor for one State and a row factor for the other), and its single readout is a *composed* State term:

    Readout("output", StateTerm("slot_value",
        Softmax(StateTerm("key_slot", scaled_query, KEY), slot), slot), dtype)

So the pipeline is: contract `key_slot` over KEY with the scaled query, softmax over the slot axis, then contract `slot_value` over the slot axis. Three consequences for a chunked implementation:

1. The kernel's written output spans the *columns* of the State it contracts (the readout lowering uses `einsum("nhf,nhfg->nhg")` for a first-Feature readout and `einsum("nhg,nhfg->nhf")` for the other). The first step above wants the column-space output of `key_slot`, which matches; the last step wants the *row*-space output of `slot_value`, which the current kernel does not produce. A row-contracting variant, or a torch contraction over a materialized chunk State, is required for the second State.
2. The softmax sits *between* the two contractions, so per-State contracted outputs are not composable by an elementwise epilogue over independent kernels: the composition is intra-chunk. This is the chunk-level form the ticket's third box is really asking for, and it is the FLA `chunk_gsa` structure (intra-chunk attention plus an inter-chunk state), not a plumbing change.
3. Because both States share one gate axis (slot) with opposite orientation, the "second factor per State axis" work from ticket 24 is a prerequisite, not an alternative.

**Honest smaller step.** The measurable win already banked for these families came from the reference path (GSA 40.0 -> 2.10 ms, rwkv6 44.7 -> 2.83 ms, hgrn 2.8 -> 0.63 ms), so the cheapest further gain is to keep improving where they actually run. The chunk-level composition deserves its own ticket, with `chunk_gsa`'s intra-chunk block as the reference algorithm and the two-State `key_slot`/`slot_value` pair as the first test case.

**Superseded checklist (kept for the record; step 2 is wrong as written).**

**Concrete target for boxes 2 and 3: `rebased`.** `examples/rebased_attention.py` exercises both remaining boxes in one Program, and it is currently rejected:

    StateSpec("numerator",   (Batch, STATE_HEADS, second_order, VALUE)),
    StateSpec("denominator", (Batch, STATE_HEADS, second_order)),      # rank-1
    # both transitions carry injections and no propagations: unit diagonal decay
    readouts=(Readout("output",
        StateTerm("numerator", query_pair, second_order)
        / (StateTerm("denominator", query_pair, second_order) + Constant(eps)),
        dtype),)

Unlike GSA, the composition is purely elementwise over the two contracted outputs (`num / (den + eps)`), so no softmax sits between the contractions and independent per-State kernels compose. Three changes, in this order:

1. `_stateful_chunked.py`: accept a rank-1 `initial_state` (`[B, H, C]`) by promoting it to `[B, H, 1, C]` with a key of ones, and take per-State tuples so the two kernels run once per State with their own carry pair. The kernels stay as they are: rows/columns are already arguments and each shape gets its own cached instance, and `rows == 1` makes the boundary pair trivial. A unit-scale State simply passes a gate of ones.
2. The lowering: a new path for "N diagonal States whose transitions have one injection, plus a readout that is an elementwise expression over per-State contractions". Per State derive the affine summary, the injection product and the readout operand, run the kernels, then evaluate the readout expression with each `StateTerm` replaced by its precomputed contracted output -- a small evaluator that leaves `Sum`/`Div`/`Constant` to torch. Gate it to contraction-only `StateTerm`s with no interior nonlinearity, which is what keeps GSA out until its intra-chunk form exists (its readout softmaxes one State's contraction before contracting the other, so that softmax would have to run over a materialized chunk State).
3. Estimate and policy: charge the aligned bound summed per State, `sum(heads * rows_i * columns_i * 4 // CHUNK_TOKENS)` per token, then re-measure `rebased` at T=512 and T=2048.

Why this is the cheapest real capability win: `based` looks similar but its readout also contains `StateTerm(name, Constant(1.0), None)` -- a whole-State reduction with no contract role -- so it needs a second kernel output form and should follow, not lead. `rebased` needs only what is listed above.

**Done: the chunked entry carries several States, one of them rank-1.** `chunked_elementwise_dense` now takes per-State tuples (gate rows/columns, key, value, query, initial State) and returns one ``(output, final)`` pair per State; the singular caller passes one-element tuples and unpacks, so the existing path is untouched. A rank-1 State is passed as ``[sequences, heads, columns]``, promoted internally to a single row, and signalled by ``key=None``, which means "this State contracts over its one row, so the key factor is the constant one"; its final State comes back at the declared rank. The test ``test_chunked_entry_runs_a_rank_one_state_beside_a_dense_one`` runs a rank-1 State and a dense one together with ``rows != columns``, against an fp32 recurrence, and then applies the quotient composition a `rebased`-shaped readout performs. Worth noting for coverage: every other case in that file uses square States, where the row and column orientations coincide.

**Correction: there is no TileLang symbol collision on this path.** An earlier entry below recorded one, from a two-State run failing with ``kernel chunk_recurrence_kernel input query shape[2] violates packed ABI constraint; expected: 5, got: 3``. That diagnosis was wrong, and what settled it is that the same two-shape probe reproduces the failure on **unmodified** code: the kernel declares ``query: (sequence_length, heads, rows)`` -- a first-Feature readout contracts the rows and writes the columns -- while the probe passed a query indexed by columns. With the query contracted over the rows, a dense State ``(rows=3, columns=5)`` and a rank-1 one ``(rows=1, columns=5)`` both ran and matched an fp32 oracle in the same process, so different-sized States coexist and the shapes are part of each kernel's cache key. The symbol hazard recorded earlier in this ticket is a different one: it needs one symbol compiled against two *output layouts*, not two States of different sizes.

**What remains for this box**, in order: gather per-State injection factors, decay summary and readout operand in the lowering; pass all States to the entry; evaluate the readout expression with each ``StateTerm`` replaced by its precomputed contracted output. Then the estimate and the policy, per State. `rebased` is still the first target.

**Superseded (kept for the record): the earlier wrong blocker note.**

**Attempted, blocked, reverted: TileLang refuses a second kernel for the same symbol.** The entry was refactored to per-State tuples and given a rank-1 promotion (a `[B, H, C]` State promoted to `[B, H, 1, C]` with `key=None` meaning "contracts over its single row, so the key factor is one"). With one State the change was inert and verified: ruff clean, 153 unit tests and both GPU chunked suites (6 tests) unchanged. With two States of *different* sizes it failed at kernel launch:

    RuntimeError: kernel chunk_recurrence_kernel input query shape[2] violates packed ABI constraint; expected: 5, got: 3

The diagnosis is the same hazard this branch hit before, now confirmed on a second path. The TileLang log shows `boundary_pair_kernel` and `chunk_recurrence_kernel` each compiled exactly **once** although two distinct `(rows, columns)` shapes were requested, so the second compilation was not used. The failing call is then validated against the *first* kernel's ABI: `got: 3` is that State's key/row-gate (`[T, H, rows]`), which no caller could have placed in a `query` slot -- the `query` passed for both States was `[T, H, columns]`, so its extent is 5 on both calls. The only argument whose third dimension is 3 is the key, i.e. a boundary call matched against the recurrence kernel.

**Consequence for the design.** Multi-State chunking is blocked on one symbol per State shape, not on the loop: `@lru_cache` keys the *Python* compile function by shape, but TileLang keys its own per-symbol cache by the kernel name, so distinct shapes collide under one symbol. Fixing it means giving each compiled kernel a shape-distinct symbol (thread the shape into the jit name, or into whatever identifies the kernel to TileLang), and that is the first thing the next attempt must settle -- before the per-State entry, since the entry is otherwise correct. The refactor itself is worth redoing in that attempt: it was inert and green with the singular caller passing one-element tuples, and the rank-1 promotion is ten lines.

The work was reverted rather than committed because the rank-1 path would otherwise be unreachable dead code -- nothing in the corpus routes a rank-1 State to this entry -- and a test cannot reach it until the symbol collision is fixed.

**Multi-State wiring attempted: the gate passes, the numbers are wrong.** Branch ``feat/chunked-multistate`` carries it, unmerged. ``execute_reference`` now early-returns into ``execute_chunked_states``; for ``rebased`` the gate passes, the per-State gathering runs and the kernels return, so the plumbing works end to end. The values do not: the FLA-parity test reports 88.1% of elements mismatched, greatest absolute difference 4.10, so the error is semantic and sits in either the gathering or the kernel's convention. ``perf/structured-lowering`` keeps only the verified entry work and its test.

Settled while wiring, and worth keeping when this resumes: the ``evaluate`` callback takes ``(expression, bound)`` because ``_injection_terms`` and ``_affine_summary`` call it that way; a whole-State ``AxisScale`` makes ``elementwise`` of shape ``[T,H,1,1]``, so the column gate has to be ``None`` rather than a slice of column 0; a rank-1 State promotes to one *column* (``[C,1]``) with its injection factor as the key, which is exactly how ``_injection_terms`` and ``_scan_state`` already present it, and the opposite of the orientation first written here; and a whole-State injection is declined because the kernels' key is separable.

The next probe, which splits the remaining error between gathering and kernel: run the operator, monkey-patch ``_stateful_chunked.execute_chunked_states`` to return ``None``, run it again, and compare -- both runs are the same Program, and the reference path is the oracle. ``examples/rebased_attention.py`` has no ``example_inputs`` helper, so the inputs come from the coverage test's own builder.

**Correction to the note above: both localizing attempts were invalid, so the mismatch is not yet attributed.** The first ran on ``perf/structured-lowering``, where ``execute_chunked_states`` does not exist, so it never exercised the branch. The second monkey-patched ``execute_chunked_states`` *after* ``_compile`` had already selected the lowering, so both runs took the same path and the comparison returned 0.0 -- which is evidence of nothing. To localize properly, compile twice: one operator compiled with the branch active and one compiled with it disabled from the start (patch before ``_compile``, or add a compile-time flag to the gate), then compare, and separately check the reference path against FLA to see whether the Program is at fault independently of chunking.

What is actually established about ``rebased`` on the branch: the gate passes, the kernels run, outputs come back, and they mismatch FLA on 88.1% of elements at rtol/atol 5e-2. The cause is open.

**Mismatch localized to the first token.** Measured on ``feat/chunked-multistate``: ``rebased`` differs from FLA already at token 0 -- max absolute difference 3.18, actual ``[0.063, -0.208, 0.159]`` against expected ``[-0.252, 0.828, -0.633]`` -- so the fault is in the first-token quantities (injection, query, or the initial State), not in the decay and not in the chunk carries. That rules out the carry/boundary machinery, which was the largest surface to suspect.

The gathered tensors themselves are consistent by construction: they come from the reference's own ``_affine_summary`` and ``_injection_terms``, called with the same arguments ``_scan_state`` passes. What stays suspect is how the entry consumes them for a 256-row State, i.e. these four lines in ``execute_chunked_states``:

    gate_rows = elementwise[:, :, :, 0]        # the row factor
    gate_columns = None                        # when elementwise.shape[3] == 1
    key = injection[:, :, :, 0]                # the row factor of the injection
    value = injection[:, :, 0, :]               # its column factor
    query = _role_vector(operand)               # over the contracted role

Next measurement, which pins the offending tensor in one shot: print those four for both States beside the reference's own ``elementwise``/``bias`` for the same inputs, and compare the token-0 result against the hand computation ``out_0 = (q_pair . key_pair_0) v_0 / (q_pair . key_pair_0 + eps)``. Note that ``rebased``'s second State is rank-1 with 256 rows, which is the shape combination the committed entry test does not cover (it uses rows=3/columns=5 and rows=1/columns=5).

**Narrowed to a scale asymmetry between the numerator and the denominator.** Three measured facts on ``feat/chunked-multistate`` with ``dim=16``, ``dim_value=64``:

1. At token 0 the expected result is exactly the value input: with ``qp = q (x) q`` and ``kp = k (x) k`` the ratio ``(qp . (kp (x) v)) / (qp . kp)`` collapses to ``v``, and an independent pure-torch oracle reproduces FLA there (max absolute 0.008, i.e. bf16 rounding) with ``eps`` of either ``1e-6`` or ``0``.
2. The chunked path returns ``[0.063, -0.208, 0.159]`` where the value is ``[-0.252, 0.828, -0.633]``: every component is the expected one times **-0.25**, and ``0.25 = dim ** -0.5``. The answer's shape is right and the scalar between numerator and denominator is wrong.
3. ``git diff perf/structured-lowering..feat/chunked-multistate -- attention_engine/`` touches only ``_stateful_chunked.py`` (+233) and the early return in ``_stateful_reference.py`` (+18), so the reference path is untouched and the error is unambiguously in the chunked path.

The suspect is the per-State query: ``queries`` come from ``term.operand``, which for `rebased` is the same ``query_pair`` expression for both States, so an asymmetry has to come from how that operand is scaled or read per State rather than from the expression. `rebased` scales its query by ``dim ** -0.5``, and a one-sided application of exactly that factor is what was measured.

**Probing note.** External patching does not reliably reach this lowering: a spy installed on ``chunked_elementwise_dense`` before ``_compile`` never fired, and patching ``execute_chunked_states`` after ``_compile`` did not change the result. The next attempt must print from inside the function.

**Delivered, `79a134a`.** `execute_reference` early-returns into `execute_chunked_states`, which runs one chunked recurrence per State and composes the readout over the contracted outputs. `chunked_states_eligible` is the static gate and also drives the plan's charge. Rank-1 States are one column per token with their injection factor as the key and `value=None` meaning the constant one, which is how `_injection_terms` and `_scan_state` already present them.

The bug that took longest is worth recording because its symptom was subtle: the kernels write `[sequence, head, token, column]` and the reference lowering's caller consumes token before head, so the raw kernel output was permuted a second time on the way out. The signature was that only element (0, 0) of every tensor survived -- head 0 token 0 matched to bf16 rounding while everything else was 84% wrong -- which is exactly what applying a permutation to an already-permuted tensor does, and it is why the injection, query, readout and kernel all measured correct while the Program output did not.

Evidence: `rebased` matches FLA; the per-State bound is published (`133120` bytes/token aligned against `25559040` per token at heads=8, dim=64; `2080` at heads=2, dim=16); the entry test covers a rank-1 State beside a dense one with `rows != columns`; the functional gate is 30 passed and the unit suite 153.

**Still open in this ticket:** the latency and envelope evidence at length for the v2/v3 families (the charge is published and T=2048 fits the envelope on paper, but the runtime acceptance and the measured latency are not yet on record), and the packed form.
