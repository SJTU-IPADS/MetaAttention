# AttentionEngine API


## Custom Attention Level API
This level API is designed for users to define their own attention mechanism. Users needs to define the following components:
- `qkv_meta`: meta information for Q, K, V, such as shape, dtype
- `score_mod`: elementwise modification on attention scores
- `mask_mod`: mask modification on attention scores
- `online_func`: online function for attention scores
- `custom_fwd_inputs`: custom inputs for attention mechanism
```py
mod = AttentionEngine(
    qkv_meta: Tuple[MetaTensor, MetaTensor, MetaTensor],
    score_mod: Callable[[Tensor, CustomIO, int, int, int, int], Tensor],
    custom_fwd_inputs: CustomIO,
    online_func: OnlineFunc,
    mask_mod: Callable[[int, int, int, int], Bool]
)
```

For compiled Attention module, users can use pytorch-compatible API to interact with the module.
```py
output = mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_inputs: Optional[List[torch.Tensor]]
)
output.backward(do)
```

### OnlineFunc

OnlineFunc is a class that defines the online function for attention scores, such as online softmax and retention.
```py
class OnlineFunc:
    def __init__(self):
        pass
    def online_fwd(scores, online_rowscales, b, h, q_idx) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        pass
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        pass
    def forward(scores, online_rowscales, b, h, q_idx, kv_idx) -> Tensor:
        pass
    def backward(dp, scores, final_rowscales, doosum, b, h, q_idx, kv_idx) -> Tensor:
        pass
```
Examples can be found in the [Getting-started Example](./getting_started_example.md).

### score_mod
`score_mod` takes the following inputs:
- `score`: attention scores
- `custom_fwd_inputs`: custom input tensors
- `b`: batch index
- `h`: head index
- `q_idx`: query index
- `kv_idx`: key index

`score_mod` returns the modified attention scores.
```
def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx) -> Tensor:
    return new_score
```

Examples can be found in the [Getting-started Example](./getting_started_example.md).

### mask_mod
`mask_mod` takes the following inputs:
- `b`: batch index
- `h`: head index
- `q_idx`: query index
- `kv_idx`: key index

`mask_mod` returns Bool value to indicate whether the attention score should be masked.
```
def mask_mod(b, h, q_idx, kv_idx) -> Bool:
    return True
```


## Stateful Operator API

Stateful sequence operators are authored as model-independent Algorithm IR and
compiled explicitly from logical metadata. Compilation accepts fixed Head and
Feature role sizes, an explicit Target, one physical mode, and independent
optimization, autotuning, and backward choices. It never accepts example
Tensors. Batch size, dense Sequence length, packed logical-sequence count,
packed token count, and offsets remain dynamic at invocation.

### Implementation directories

The public interface remains in `attn_engine/stateful_operator.py`, with the
existing exports from `attn_engine`. Private implementation modules live under
`attn_engine/stateful/`:

| Directory | Responsibility |
| --- | --- |
| `analyse/` | Algorithm IR structural validation, typing, simplification, Scanable proofs, and Backward IR derivation |
| `compile/` | Mathematical plans, Lowered candidates, workspace estimates, selection, tuning, and benchmark workloads |
| `runtime/` | Invocation validation and logical-role/physical-tensor bindings |
| `lowering/` | Structured proofs, recipes, schedules, execution, and backward transport |
| `lowering/{expression,diagonal,scalar,lowrank,scan,readout}/` | Function-local lowering implementations and kernels |

Lowering-specific recipe compilation and execution stay together for locality;
the directory move does not split their interface or introduce a new seam or
adapter. The old private `_stateful_*` import paths and the forwarding
`_stateful_tilelang` module are removed without compatibility aliases. Kernel
arithmetic and compiler-owned cache identities are unchanged; backend JIT
artifacts may be regenerated under the relocated module names.

All Stateful kernel families now use released TileLang `0.1.15`, including
generated expressions, diagonal/native readouts, scalar/vector scans, low-rank
WY, and dense/packed checkpoint forward/backward. Kernel binaries remain
shape-specialized; dynamic public invocation sizes do not imply one symbolic
Batch/Sequence binary.
This correctness-first cutover does not promise the historical timings below.


```py
from attn_engine import StatefulOperator, Target

operator = StatefulOperator(algorithm)
compiled = operator.compile(
    role_sizes={
        query_heads: 2,
        key_heads: 2,
        value_heads: 2,
        state_heads: 2,
        key_dim: 64,
        value_dim: 128,
    },
    target=Target(
        # Optional compiled envelope; larger runtime workloads fail explicitly.
        workspace_bytes=1 << 30,
    ),
    mode="dense",
    optimized=True,
    autotune=False,
    backward=True,
)
result = compiled(
    query=query,
    key=key,
    value=value,
    gate=gate,
    return_final_state=True,
)
output = result.outputs["output"]
state = result.final_state["memory"]
```

One dense compiled callable accepts different positive Batch and Sequence
sizes. A packed callable uses `mode="packed"`; invocation supplies contiguous
token-major inputs and a device-local contiguous int32 `sequence_offsets`
vector. Different offsets shapes and values, including empty sequences and
all-empty workloads, reuse the same artifact. Runtime tensors must preserve the
compiled role sizes, Algorithm IR dtypes, Target, mode, and canonical contiguous
layout. No implicit copy, transpose, dtype conversion, device move, replanning,
or recompilation occurs.

Every invocation returns immutable `ExecutionResult`, containing immutable
named outputs and optional named final State. Missing initial State means one
FP32 zero State per logical sequence. Final State is materialized only when
requested. Calling an uncompiled `StatefulOperator` raises; only the callable
returned by `compile()` executes.

With `backward=False`, grad-enabled invocation is rejected. With
`backward=True`, ordinary PyTorch autograd transports cotangents and saved
context through the compiler-selected forward/backward composition. The
compiler owns Backward IR and context semantics; no eager or framework-derived
fallback is used.

The first schema has one named matrix State with two ordered Feature roles. It
supports typed Input Expressions (`Input`, BF16/FP32 `Constant`, `Add`,
`Multiply`, `Negate`, `Exp`), role-directed Axis Scale, general Rank-One
propagation, Product Injection tuples, explicit Head Mapping, and single-role
State Contractions. Compiler analysis, Mathematical Plans, Lowered Candidates,
KernelConfig, context contracts, and caches are internal.

Schema v2 keeps every v1 form and adds multiple States, rank-1 States,
expression readouts, token shifts, and typed arithmetic needed by the Stateful
Operator examples.

- **Several independent States** with one Transition each; States share one Head
  role. The v2 vocabulary used at most two distinct Feature roles across the
  Program; v3 removes that Program-wide limit, while each State still has one
  or two Feature axes.
- **Rank-1 States** (`Batch, Head, Feature`) for scalar accumulators such as a
  linear-attention denominator.
- **Expression readouts**: `Readout(name, expression, output_dtype)` where the
  expression may combine `StateTerm` contractions, inputs, and arithmetic,
  including `Divide` by a denominator State, `Clamp`, and a non-State additive
  term. A `StateTerm` with `role=None` reads a rank-1 State without contraction.
- **More Input Expressions**: `Divide`, `Log`, `Sigmoid`, `Sqrt`, `Reciprocal`,
  `Clamp`, and `Shift(expr, lag)` for token shift. Transcendental and reciprocal
  forms compute in FP32, as `Exp` already did.
- **Independent States stay Scanable**: a Transition is driven by Inputs only,
  so the per-token summary remains affine and the parallel scan still applies.

Schema v3 keeps every v2 form and adds the vocabulary the product-Feature
families need (`docs/adr/0003-product-features-and-prior-state-readouts.md`):

- **Product Feature roles.** `ProductFeatureRole(left, right)` declares a Feature
  whose axis is the flattened product of two Features (named `left_x_right`); a
  State may use it as an axis, and `Outer(left, right, role)` builds that axis
  from two expressions. Based and Rebased keep their `[K*K, V]` numerator as an
  ordinary matrix State.
- **Feature reductions.** `Contract(operand, coefficient, role)` reduces one
  Feature axis and keeps a singleton axis, so a Program can write a coefficient
  term such as RWKV6's `(q * u * k)` bonus. Its `coefficient` may be feature-free.
- **Axis softmax.** `Softmax(operand, role)` normalizes one Feature axis, which is
  how a Program joins two States through a slot distribution (Gated Slot
  Attention) rather than through a fixed contraction.
- **Prior-State readouts.** `StateTerm(..., prior=True)` reads the State before
  the current token's transition, which pre-transition readouts such as RWKV6
  require.
- **Feature-free Injection factors.** An Injection factor may declare a Feature
  role its expression does not carry; the value broadcasts along that axis, so a
  constant `+1` Injection expresses a token count.

Schema-v2 Programs enter the reference lowering in
`attention_engine/attn_engine/stateful/lowering/reference.py`. Dense inference whose
typed transitions are diagonal, with disjoint injection factors and supported
expression readouts, uses FP32 chunk summaries, boundary carries and independent
64-token replay. Identity propagation uses tiled GEMMs; Product Feature operands
are synthesized inside tiles rather than stored as token-sized outer products.
Prior-State terms and readout dependencies retain their authored semantics.
Packed and gradient execution retain the segmented affine scan. Programs within
the v1 language retain typed specialized TileLang plans.

Stateful Operator examples keep each per-token recurrence in its own
`examples/<family>.py` module so the examples exercise the compiler directly.

Schema v4 adds ordered affine steps and coupled State updates:

- `StateTransition("memory", steps=(...))` interleaves `AxisScale`,
  `RankOnePropagation`, `ProductInjection` and `StateInjection`. Each step acts
  on the running target State. Use `steps` or the existing grouped
  `propagations/injections` spelling; mixing them is an error.
- `StateReference("memory", prior=True)` selects the State before this token.
  The default selects another State's completed Current State. Current
  references induce a Transition DAG, sorted independently of author order;
  cycles, including current self references, fail with `IR_DEPENDENCY`.
- `StateInjection(source, coefficient)` adds a scaled source State.
  `StateInjection(source, coefficient, role=KEY, left=k, right=p)` adds
  `coefficient * k * (p^T source)` along KEY. It has no implicit identity term.
  Its coefficient and vectors must be State-free, preserving joint affinity.

The first coupled lowering requires all States to share ordered logical roles,
with all non-scalar actions on one shared Feature axis. It stacks States along
that axis, composes the token-local DAG's block-affine map, and scans the sequence
in parallel. It supports dense/packed execution, initial/final States, prior
Readouts and gradients through the reference scan. Workspace is charged for the
joint axis. Momentum Delta Rule is an example of this path; no performance
parity is claimed. Delta Product uses ordered steps and retains the compiled
factored path by transforming write factors through later actions.

State Terms remain reserved for Readouts within an affine AlgorithmIR.
Schema v5 permits nonlinear State-derived coefficients through an explicit
Scan Stage DAG, described below. Readout `prior=True` retains its existing
meaning. Intermediate target values are implicit in step order; arbitrary
named intermediate State references are not introduced. `Shift` zero-fills
invocation sequence starts and does not carry input history across calls.
Product Injections may contain several typed factors; some specialized
lowerings require two vector factors.

See [ADR 0004](adr/0004-ordered-affine-state-transitions.md) and
[the domain glossary](../GLOSSARY.md).

The first production Target is CUDA H20. It accepts BF16/FP32 inputs and
outputs, FP32 State, canonical contiguous layouts, and Feature pairs in
`{64,128} × {64,128}`. Invalid IR, unsupported composition, Target mismatch,
invalid packed offsets, unsupported specialization, resource failure, and
codegen failure surface as structured `StatefulCompilationError` values.
ADRs 0002 and 0003 record the implemented language extensions described above.

### Scan Stage DAG (schema v5)

`StagedStatefulOperator(ScanStageIR(stages, outputs))` composes separately
scanable programs. A `ScanStage(name, algorithm, bindings=())` contains an
`AlgorithmIR` or `LogAddExpScanIR`. `StageBinding(input, stage, output)` binds a
downstream declared TensorInput to an upstream named Readout; only unbound
inputs appear in the public call. `StageOutput(name, stage, output)` selects a
public output. The compiler checks matching logical roles/dtype, permits
physical role permutations, sorts stage dependencies and rejects cycles with
`IR_DEPENDENCY`. Bindings do not insert implicit dtype casts.

State names are globally unique. Public `StateTuple` order follows stage
**declaration** order and each child's State order, even when topology changes
the execution order. All stages share one invocation's sequence/offsets and
Batch count; each retains its own State Head role. A downstream HeadMapping
can expand a key-head trajectory to value heads without duplicating the public
auxiliary State. Stage bindings export scalar/vector readouts; affine matrix
trajectories are not an additional public binding form.

```python
from attn_engine import Target
from examples.preconditioned_delta import (
    KEY, VALUE, KEY_HEADS, VALUE_HEADS, preconditioned_delta,
)

compiled = preconditioned_delta(kind="gdn").compile(
    role_sizes={KEY_HEADS: 2, VALUE_HEADS: 4, KEY: 64, VALUE: 64},
    target=Target(), mode="dense", backward=True, checkpoint=True,
)
# Invoke with named query/key/value, log_atk_decay, beta_atk, center,
# scale, log_decay and beta. StateTuple names are ("atk", "memory").
# compiled.workspace_bytes(B*T, sequence_count=B, sequence_length=T)
# reports the conservative staged resource envelope.
```

`Abs(operand)` / `expr.abs()` computes in FP32 with zero subgradient at zero.
`Cast(operand, dtype)` / `expr.to(dtype)` explicitly places an FP32 or BF16
conversion in the expression DAG. It preserves roles, and its position matters
for rounding (Cast before multiplication can differ from Cast after it).

`LogAddExpScanIR(logits: TensorInput, state: StateSpec | None)` defines only
`z_current = logaddexp(z_prior, logits)`, defaulting to State name `z` and
identity `-inf`. Inputs and explicit initial State accept finite values and
`-inf`; NaN/+inf fail the runtime algebra check. Its FP32 Readouts are
`normalized_key = exp(logits-z_current)` and `log_gate`; gate is
`z_prior-z_current` for finite prior and 0 for cold `-inf` prior. An all-masked
identity element produces zero key/gate. The identity/identity VJP is zero.
`LogAddExpOperator(ir).compile(...)` also supports standalone invocation.
Use [stable LightNet](../examples/lightnet_stable.py) to combine this stage
with an affine matrix memory; its public State order is `(z, memory)`.
FLA uses `(memory, z)` with z shaped `[B,1,H,K]`; engine z is `[B,H,K]`.

Stages execute existing child lowerers and materialize only retained outputs.
Torch autograd chains downstream coefficient gradients back through upstream
expressions, trajectories and initial States. `checkpoint=True` recomputes
child intermediates using non-reentrant checkpoint. The resource estimate
remains conservative and charges the non-checkpoint scan levels; checkpoint
reduces retained tensors, not the admitted resource envelope. Packed includes
empty and all-empty sequences. Dense affine stages require positive length,
as before; standalone LogAddExp additionally supports dense length zero.

Preconditioned KDA/GDN and stable LightNet have independent reference, State,
continuation and gradient checks. Their current staged execution is a
capability baseline; generic scan storage and public-call latency still need
performance work. See [ADR 0005](adr/0005-scan-stage-dag-and-logaddexp.md) and
[stage benchmarks](stateful-stage-benchmarks.md) for measured limits.

### Structural inference schedules

Identity and scalar propagation use the existing fused scalar-factorized dense
kernel. A previously measured aligned scalar chunk schedule improved CUDA Graph
replay but regressed public-call latency; scalar inference now consistently uses
the raw dense kernel rather than that multi-launch schedule.

Diagonal multi-State inference supports batches, nonaligned tails, nonzero
initial States, FP32 final States and continuation. Readout expressions consume
contracted FP32 terms without materializing per-token matrix States. Unsupported
typed structures retain the generic lowering; selection does not inspect model
or example names.

Ordered first-axis Rank-One propagations with row, column or whole-State scales
use an FP32 TileLang recurrence, preserving authored operation order, multiple
injections, independent first-axis contractions, and final-State ownership.
Dense/packed training uses chunk summaries, boundary carries, replay, local VJPs
and deterministic reductions in the checkpoint lowerer. Unsupported gradient
compositions retain the generic lowering.

When every ordered propagation is a whole-State scalar scale or first-axis
Rank-One update, at most four rank/injection events and dimensions at most 128
use the compact TileLang WY triangular solve and event replay. Factors are
bound through their strides and packed into FP32 operation-kind buffers;
multiple injection/readout factors are stacked. This changes preparation cost
relative to the previous Triton schedule. Zero/signed gates use direct interval
products, not prefix inverses. Distinct left/right factors, nonzero initial
States and continuation retain the same recurrence. Noncommuting row/column
scales use the ordered recurrence; packed and backward use checkpoint kernels.

The workspace envelope charges token factors and expression outputs separately
from chunk carries: `B*T*bytes_per_token + B*(ceil(T/64)+1)*bytes_per_chunk`.
The extra chunk allowance covers final States. The runtime checks this bound
before launching; it no longer allocates an unused duplicate reservation buffer.

`testing/benchmark_stateful_plans.py --families based,rebased,rwkv6,gsa,hgrn,gdn,rwkv7,pressure --seqlen 512 2048 --json /tmp/plans.json`
reports the selected lowerer, public-call timing, CUDA Graph replay, launches,
and workspace. Correctness uses the independent authored-order recurrence.

Measured on an idle H20, BF16, `B=1,H=8,T=2048,K=V=64`, five warmups and
median of twenty CUDA-event samples around the public callable with final State:

| Structure/example | Before end-to-end ms | After end-to-end ms | After Graph ms |
| --- | ---: | ---: | ---: |
| Based (`K=16`) | 11.384 | 1.931 | 0.816 |
| ReBased (`K=16`) | 2.814 | 1.084 | 0.609 |
| Normalized linear | 0.766 | 0.874 | 0.203 |
| RWKV6 | 2.796 | 0.908 | 0.365 |
| GSA (`slots=16`) | 2.119 | 0.988 | 0.312 |
| HGRN (`H=1`) | 0.570 | 0.519 | 0.060 |
| GDN | 4.661 | 1.074 | 0.604 |
| RWKV7 State | 4.968 | 1.031 | 0.601 |
| Ordered composition pressure | 82.628 | 1.372 | 0.924 |

The results pass an independent FP32 token recurrence for outputs and final
States at lengths 65, 512, and 2048, including nonzero initial States and
batched continuation splits.

The historical follow-up Triton packing pass reused single injection/readout
bindings and left unused operation buffers at one element. On that H20 workload, the
`T=2048` graph timings changed from `0.792` to `0.712 ms` for Based and from
`0.600` to `0.566 ms` for ReBased; profiler-attributed auxiliary launches fell

from `71` to `47` and `33` to `21`, respectively. GDN changed from `0.598` to
`0.592 ms` graph time. Output, final State, and continuation correctness remain
covered by the focused GPU suite.

The compiler-only compositional comparison was measured on an otherwise idle H20
with BF16 inputs, FP32 gates and State, `B=1`, `(row,column)=(64,128)`, `T=129`,
one public call with final State, three warmups, five CUDA-event samples per batch,
and ten alternating batches per side. The generic side used the same Algorithm IR
and `optimized=False`; compilation was excluded. Paired batch medians were:

| Composition | Optimized lowerer | Generic CUDA-event ms | Optimized CUDA-event ms | Generic / optimized | Optimized peak / generic peak |
| --- | --- | ---: | ---: | ---: | ---: |
| Axis scales | `tilelang-chunked-stateful` | 5.106 | 0.233 | 21.95x | 0.28% |
| Axis + Rank-One | `tilelang-factored-recurrence` | 5.319 | 0.778 | 6.83x | 1.88% |
| Multiple injections | `tilelang-product-gemm` | 0.744 | 0.761 | 0.98x | 100.42% |
| Rank-One | `tilelang-factored-recurrence` | 5.131 | 0.543 | 9.46x | 1.41% |

The multiple-injection candidate is semantically valid and stays within the
predeclared 1.10x latency and memory upper-bound gate: its optimized/generic
CUDA-event ratio is 1.023 with a bootstrap 95% interval of [1.006, 1.050], and
its optimized/generic incremental-allocation ratio is 1.004. It is 2.3% slower
than generic at this workload, so the measurement supports retaining the
candidate for semantic coverage, not claiming a speedup. Full raw records are
in `benchmark_results/composition_profiles_final.json`; this artifact is local
measurement evidence, not a tracked API contract.

The subsequent multiple-injection pass admits matrix diagonal transitions with
several disjoint row/column Product Injections when all factors and readouts have
native raw-loader proofs. The compiler retains an ordered tuple of factor pairs;
each replay token sums the FP32 injection products before adding them to propagated
State. Authored expression association and BF16 node rounding remain in the loaders.
Chunk summaries and all readouts share one State carry. No full token-sized State
history, grouped-head expansion, or promoted input copies are materialized. The
single-injection normalized and shared-score schedules are not used for this case;
unsupported expressions, packed execution and gradients retain their existing paths.
Eligibility and candidate pricing depend on typed structure and fixed Head/Feature
sizes, including injection count, never input names or runtime B/T.
The loader also preserves FP32 zero-dimensional constants through arithmetic,
rounding the result at each BF16 expression node rather than first quantizing the
constant to BF16. Non-binary constants such as `0.3` exercise this distinction.

On an idle H20, the same-IR public-call comparison at `B=1,T=129`, four State heads,
`(row,column)=(64,128)`, BF16 inputs and FP32 State used ten alternating batches,
ten warmups and twenty samples per batch, with final State requested:

| Path | CUDA-event ms | Wall ms | Peak incremental allocated bytes |
| --- | ---: | ---: | ---: |
| Generic (`optimized=False`) | 0.750024 | 0.774290 | 93,401,088 |
| Multiple-injection chunked | 0.286528 | 0.304401 | 926,720 |

This is 2.62x generic/chunked CUDA-event latency and 0.992% of generic's incremental
allocation. The paired chunked/generic batch-median ratio has a bootstrap 95%
interval of `[0.373,0.390]`. Records are in
`benchmark_results/multiple_injections_after.json`; the separate earlier baseline
in `benchmark_results/multiple_injections_before.json` is not mixed into the
same-process speedup calculation. These are local measurement artifacts.

Final-source verification passed 192 CPU tests (three skipped) and 162 focused H20
tests. A single compiled handle per structure covered two and three injections,
identity/row/dual propagation, unequal Features, grouped heads, a headless FP32
factor, fresh storage-offset tensors, signed/zero gates, `T={1,63,64,65,137,4097}`,
nonzero initial State, FP32 post-State and BF16 prior-State outputs, output-only,
and continuation against an independent CPU recurrence without widening tolerance.
Obsolete lowerer-name assertions were removed; the older dependent-readout fixture
now invokes the public callable rather than passing normalized values to a raw
executor.

Fresh-process public and CUDA Graph smoke reused one handle across runtime workloads.
Ten alternating batches with ten warmups and ten samples per batch gave these
within-run comparisons; compilation was excluded:

| B | T | Generic public ms | Chunked public ms | Generic / chunked | Chunked Graph ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.691392 | 0.185408 | 3.73x | 0.058592 |
| 2 | 65 | 0.726256 | 0.279312 | 2.60x | 0.107248 |
| 1 | 129 | 0.686864 | 0.279768 | 2.46x | 0.107808 |
| 2 | 257 | 1.026528 | 0.302328 | 3.40x | 0.140416 |

All outputs, final States, output-only calls, continuation and Graph replay passed
the independent oracle. Generic `B=2,T=512` and `T=2048` were rejected by its
existing 1 GiB workspace envelope; native correctness passed before those generic
measurements. No speedup ratio is assigned to those rejected calls. Evidence:
`benchmark_results/multiple_injections_final.json` (local artifact).



The follow-up dense-State pass also reuses identical broadcast value bindings
across States; it changes no numerical path and is covered by the same focused
GPU regression suite.

Dense inference selected as `chunked-stateful` or `product-feature-chunked`
now enters a direct executor with compile-time State/readout/layout recipes,
without constructing reference sequence segments or repeating eligibility
analysis. Reference execution remains the semantic path for unsupported
structures, packed programs, gradients, and unoptimized v2 programs.

The rank-1 diagonal/identity-readout structure (including HGRN) uses raw BF16/FP32
bindings and native vector affine scans. A sequence of at most 64 tokens uses
one output kernel; longer sequences use summary, carry, and output kernels.
Three FP32 chunk vectors are scratch; beyond 64 chunks, recursive group summaries
bound prefix depth and add smaller vector buffers. No promoted matrix, dense query
of ones, FP32 input copy, or token-sized State scratch is created. Output casting
and optional FP32 final-State writing occur in the output kernel. HeadMapping,
nonzero initial State, tails, zero/signed gates and continuation are preserved.

On H20, the HGRN matrix `B={1,2}`, `H={1,8,16}`,
`T={63,64,65,130,2048,8192}` and all three contracts produced 108 runnable
records. Of 106 before/after pairs, public latency geometric means decreased
by 82.5% (output-only), 82.2% (final-state), and 80.2% (continuation), with no
observed paired regression. Two B=2/H=16/T=8192 calls previously rejected by
the old resource estimate now run with the native workspace bound. At B=H=1,
T=64 output-only changed from 0.485 to 0.122 ms and CUDA activity count from

Matrix diagonal inference now binds authored pointwise expressions directly to
raw BF16/FP32 inputs. Row and column summary scales occupy separate vectors,
not a broadcast matrix. Prefix composition uses multiplication/addition only,
including zero or signed gates, and recursively groups long chunk sequences.
One-chunk calls bypass summary/carry allocation; output-only calls omit the last
summary because no following chunk or final State consumes it.

Readout schedules are compiler-selected structural candidates: bounded chunk
replay, shared score generation followed by block contractions when the column
action is identity, and a two-State contraction/slot-softmax/contraction region
with an explicit materialization-cut alternative. Scores are reused across equal
State/query/prior/bonus expressions. Softmax reduces the typed Feature axis, not
Sequence. Fusion eligibility uses the existing padded-axis/live-FP32 resource
bound. Candidate feasibility, tuning allocation checks, and runtime workspace
policy consume the same recipe-derived output and scratch estimates, including
every output dtype and final-State allowance. Dynamic B/T do not select a new
plan. Successful device constraints are cached per process/device/constraint;
tensor binding, layout, alias, and shape checks still run on each invocation.

Shared scores were slower than replay on the measured RWKV6 shape and are not
enabled by default; fusion likewise remains a candidate, not a universal rule.
The native replay/fused defaults passed independent CPU FP32 recurrences at
`B=2,H=8,T={1,63,64,65,130,2048,8192}` for RWKV6, GSA, and HGRN, including nonzero
initial States, output-only, all final States, and nonaligned continuation.
Focused public-call regressions cover unequal Feature widths, shared heads,
BF16 authored expression rounding, zero/signed gates, prior-State reads, shared
scores, and the fused/cut softmax alternatives.
Verification: 187 unit tests passed (3 GPU-marked tests skipped), 113 focused
GPU tests passed, and the additional rank-1 autotune regression passed. RWKV6,
GSA, and HGRN autotune invocations also passed output and
final-State recurrence checks. At `B=2,H=8,T=8192`, allocator peaks for replay,
shared-score, fused/cut, and vector paths all stayed below their compiled bounds.

Final H20 measurements use BF16, `B=1`, output-only, five warmups and twenty
samples. Public-call medians below include CPU dispatch; CUDA Graph replay does
not. Full records: `benchmark_results/diagonal_native_final.json`.


The subsequent host-overhead pass compiles each native dense input's
`(Batch, Head, Sequence, Feature)` axis map once. Vector, normalized-pair, and
matrix-diagonal loaders reuse that immutable recipe rather than constructing a
full `BindingDescriptor` or searching logical roles on every call. The general
descriptor/reference interface is unchanged. Recipes retain only input names
and axis positions, never input tensors, pointers, strides, dynamic B/T, or
cross-call validation results. Current tensor strides and head extents are read
at each invocation, after the existing public runtime checks.

Same-process H20 measurements, BF16, `B=H=1`, output-only, ten warmups and median
of one hundred public-call samples; kernels and workspace policy are unchanged:

| Structure | T | Before public ms | After public ms | After Graph ms |
| --- | ---: | ---: | ---: | ---: |
| HGRN | 64 | 0.118 | 0.088 | 0.013 |
| HGRN | 2048 | 0.169 | 0.138 | 0.019 |
| RWKV6 | 64 | 0.261 | 0.169 | 0.064 |
| RWKV6 | 2048 | 0.355 | 0.248 | 0.083 |
| GSA | 64 | 0.326 | 0.241 | 0.122 |
| GSA | 2048 | 0.474 | 0.369 | 0.136 |
| Normalized linear | 64 | 0.158 | 0.120 | 0.015 |
| Normalized linear | 2048 | 0.227 | 0.175 | 0.025 |

Graph replay times remain effectively unchanged. A 500-call HGRN host profile
dropped from 227501 to 166501 function calls; full descriptor construction no
longer appears on its raw-input path. Records:
`benchmark_results/dense_raw_binding_final.json`. This removes repeated host
structure work, not GPU arithmetic, and does not establish cross-device speedup.
Verification: 330 unit/focused GPU tests passed. A fresh-process smoke reused one
compiled handle per structure across `B/T={(1,1),(2,65),(1,130),(2,8192)}`, `H=8`,
checking independent FP32 outputs, all final States, output-only, nonaligned
continuation, and dtype rejection after warm calls. The permanent vector test
also changes input storage offsets, uses a headless gate, and retains earlier
outputs to catch stale pointers/strides or accidental output reuse.

The next host-validation pass compiles immutable structural validation metadata
with the specialization. It stores input names, declaration roles, dtype and
differentiability, dense/packed Sequence positions, State role suffixes,
HeadMapping, and the reference-lowering classification. Runtime validation still
checks the current tensors, dynamic B/T or packed offsets, device, layout, dtype,
gradient contract, aliases, fixed role sizes, state tensors, and workspace.
No tensor, pointer, stride, offset value, or successful runtime validation result
is retained.

On the same H20 setup, BF16 `B=H=1,T=2048`, ten warmups and one hundred public
samples, the current public medians were HGRN `0.116 ms`, RWKV6 `0.217 ms`, GSA
`0.331 ms`, and normalized-linear `0.148 ms`; Graph medians were respectively
`0.018`, `0.081`, `0.134`, and `0.024 ms`. A 500-call HGRN cProfile decreased
from `175001` to `156001` function calls, with `_validate_runtime` cumulative
time decreasing from `0.025 s` to `0.018 s`. These are host-structure measurements
and do not imply a GPU arithmetic change.

# Upcoming Features

## Attention Library Level API
This level API is designed for users to use the existing attention mechanism in the library.
```py
mod = AttentionLibrary(
    attn_type: str="SoftmaxAttention",
    mask_type: str="Causal",
    use_types: str="Train",
)
```

## Custom Attention Level API
- Support for varlen, block-sparse mask and block-sparse indices
```py
mod = AttentionEngine(
    ...
)
mod(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_len_q: torch.Tensor=None,
    cu_len_kv: torch.Tensor=None,
    block_sparse_mask: torch.Tensor=None,
    block_sparse_indices: torch.Tensor=None,
    block_num: torch.Tensor=None,
)
```
- Support OnlineFunc for decoding
```py
class OnlineFunc:
    ...
    def combine(final_rowscales)-> Tensor:
        """Compute logic for the combine kernel"""
        return o_scale

```
- Support mask_mod for decoding

```py
def mask_mod(b, h, q_idx, kv_idx, custom_fwd_inputs) -> Bool:
    """The offset of q need to be passed by custom_fwd_inputs"""
    return True
```
