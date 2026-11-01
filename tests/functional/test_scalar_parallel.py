"""Scalar/identity dense schedules against an independent token recurrence."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Sequence,
    StateContraction,
    StateSpec,
    StateTransition,
    StateTuple,
    StatefulOperator,
    Target,
    TensorInput,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]

KEY = FeatureRole("scalar_test_key")
VALUE = FeatureRole("scalar_test_value")
QUERY_HEADS = HeadRole("scalar_test_query_heads")
KEY_HEADS = HeadRole("scalar_test_key_heads")
VALUE_HEADS = HeadRole("scalar_test_value_heads")
STATE_HEADS = HeadRole("scalar_test_state_heads")


def _compile(*, heads, key_heads, value_heads, rows, columns, dtype, gate_kind):
    inputs = [
        TensorInput("query", (Batch, QUERY_HEADS, Sequence, KEY), dtype),
        TensorInput("key", (Batch, KEY_HEADS, Sequence, KEY), dtype),
        TensorInput("value", (Batch, VALUE_HEADS, Sequence, VALUE), dtype),
    ]
    propagations = ()
    if gate_kind is not None:
        inputs.append(
            TensorInput("gate", (Batch, VALUE_HEADS, Sequence), torch.float32)
        )
        gate = Input("gate")
        propagations = (AxisScale(Exp(gate) if gate_kind == "log" else gate),)
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=tuple(inputs),
            states=(StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),),
            transition=StateTransition(
                "memory",
                propagations=propagations,
                injections=(
                    ProductInjection(
                        (
                            ProductFactor(Input("key"), (KEY,)),
                            ProductFactor(Input("value"), (VALUE,)),
                        )
                    ),
                ),
            ),
            readouts=(
                StateContraction(
                    "output", "memory", Input("query") * rows**-0.5, KEY, dtype
                ),
            ),
            head_mapping=HeadMapping(
                {
                    QUERY_HEADS: STATE_HEADS,
                    KEY_HEADS: STATE_HEADS,
                    VALUE_HEADS: STATE_HEADS,
                }
            ),
        )
    )
    return operator.compile(
        role_sizes={
            QUERY_HEADS: heads,
            KEY_HEADS: key_heads,
            VALUE_HEADS: value_heads,
            STATE_HEADS: heads,
            KEY: rows,
            VALUE: columns,
        },
        target=Target(),
        mode="dense",
        optimized=True,
    )


def _inputs(
    batch,
    heads,
    key_heads,
    value_heads,
    length,
    rows,
    columns,
    dtype,
    gate_kind,
    device,
):
    generator = torch.Generator().manual_seed(31 + batch + length)
    inputs = {
        "query": torch.randn(batch, heads, length, rows, generator=generator).to(dtype),
        "key": torch.randn(batch, key_heads, length, rows, generator=generator).to(
            dtype
        ),
        "value": torch.randn(
            batch, value_heads, length, columns, generator=generator
        ).to(dtype),
    }
    if gate_kind is not None:
        gate = -torch.rand(batch, value_heads, length, generator=generator) * 0.05
        gate[:, :, :2] = 0.0
        for edge in range(64, length, 64):
            gate[:, :, edge - 1 : edge + 2] = 0.0
        if gate_kind == "direct":
            gate = gate.exp()
            # Products must remain valid across zero and negative factors. The
            # reset is deliberately before/at/after a chunk boundary.
            gate[:, :, 30] = 0.0
            gate[:, :, 31::17] *= -1.0
            gate[:, :, 63:66] = 0.0
        inputs["gate"] = gate
    initial = torch.randn(batch, heads, rows, columns, generator=generator) * 0.2
    return {name: tensor.to(device) for name, tensor in inputs.items()}, initial.to(
        device
    )


def _reference(inputs, initial, gate_kind):
    # This oracle does not use a chunk summary, prefix ratios, or the runtime's
    # lowering helpers. Each authored token transition writes FP32 State.
    query = inputs["query"].cpu().float()
    heads = query.shape[1]
    key = inputs["key"].cpu().float()
    value = inputs["value"].cpu().float()
    key = key.repeat_interleave(heads // key.shape[1], dim=1)
    value = value.repeat_interleave(heads // value.shape[1], dim=1)
    gate = None
    if gate_kind is not None:
        gate = inputs["gate"].cpu().float()
        gate = gate.repeat_interleave(heads // gate.shape[1], dim=1)
        if gate_kind == "log":
            gate = gate.exp()
    state = initial.cpu().clone()
    output = []
    for token in range(query.shape[2]):
        if gate is not None:
            state = gate[:, :, token, None, None] * state
        state = state + key[:, :, token, :, None] * value[:, :, token, None, :]
        output.append(
            torch.einsum(
                "bhr,bhrc->bhc", query[:, :, token] * query.shape[-1] ** -0.5, state
            )
        )
    return torch.stack(output, dim=2).to(inputs["query"].dtype), state


def _assert_result(result, expected, expected_final):
    torch.testing.assert_close(
        result.outputs["output"].cpu(), expected, rtol=0.02, atol=0.05
    )
    assert result.final_state is not None
    assert result.final_state["memory"].dtype == torch.float32
    torch.testing.assert_close(
        result.final_state["memory"].cpu(), expected_final, rtol=0.01, atol=0.02
    )


def test_candidate_replacement_rebinds_execution_instead_of_retaining_scalar_adapter(
    monkeypatch, gpu_device
):
    from dataclasses import replace

    from attn_engine.stateful.compile.plan import TileLangExecutable
    from attn_engine.stateful.runtime.workspace import WorkspacePolicy

    compiled = _compile(
        heads=1,
        key_heads=1,
        value_heads=1,
        rows=64,
        columns=64,
        dtype=torch.float32,
        gate_kind="log",
    )
    inputs, initial = _inputs(1, 1, 1, 1, 65, 64, 64, torch.float32, "log", gpu_device)
    expected, expected_final = _reference(inputs, initial, "log")
    owner = compiled.operator
    executable = owner._plans[compiled.specialization]
    assert executable.runtime_family == "scalar-factorized"
    raw = TileLangExecutable.scalar_factorized_dense_raw
    calls = []

    def record(self, *args, **kwargs):
        calls.append(self.plan.candidate.identity)
        return raw(self, *args, **kwargs)

    monkeypatch.setattr(TileLangExecutable, "scalar_factorized_dense_raw", record)
    kwargs = {
        "initial_state": StateTuple(("memory",), (initial,)),
        "return_final_state": True,
    }
    _assert_result(compiled(**inputs, **kwargs), expected, expected_final)
    assert calls == [executable.plan.candidate.identity]
    generic = next(
        item
        for item in executable.plan.lowered_candidates
        if item.lowerer == "tilelang-generic"
    )
    estimate = dict(executable.plan.candidate_estimates)[generic.identity]
    replacement = replace(
        executable,
        plan=replace(
            executable.plan,
            candidate=generic,
            mathematical_plan=next(
                item
                for item in executable.plan.mathematical_plans
                if item.identity == generic.plan_identity
            ),
            workspace=WorkspacePolicy.for_candidate(
                generic,
                estimate,
                compilation=compiled.compilation,
                algorithm=owner._program,
                rank_one_factored=None,
            ),
        ),
    )
    owner._plans[compiled.specialization] = replacement
    _assert_result(compiled(**inputs, **kwargs), expected, expected_final)
    assert calls == [executable.plan.candidate.identity]
    owner._plans[compiled.specialization] = executable
    _assert_result(compiled(**inputs, **kwargs), expected, expected_final)
    assert calls == [executable.plan.candidate.identity] * 2


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("gate_kind", [None, "log"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dynamic_scalar_lengths_final_state_continuation_and_graph(
    batch, gate_kind, dtype, gpu_device
):
    """One public executable crosses streaming, chunk-grid and tail workloads."""
    heads, rows, columns = 2, 64, 64
    compiled = _compile(
        heads=heads,
        key_heads=heads,
        value_heads=heads,
        rows=rows,
        columns=columns,
        dtype=dtype,
        gate_kind=gate_kind,
    )
    for length in (63, 64, 65, 130, 2048):
        inputs, initial = _inputs(
            batch,
            heads,
            heads,
            heads,
            length,
            rows,
            columns,
            dtype,
            gate_kind,
            gpu_device,
        )
        initial_before = initial.clone()
        expected, expected_final = _reference(inputs, initial, gate_kind)
        with torch.no_grad():
            result = compiled(
                **inputs,
                initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
            output_only = compiled(
                **inputs, initial_state=StateTuple(("memory",), (initial,))
            )
        _assert_result(result, expected, expected_final)
        assert output_only.final_state is None
        torch.testing.assert_close(
            output_only.outputs["output"].cpu(), expected, rtol=0.02, atol=0.05
        )
        torch.testing.assert_close(initial, initial_before, rtol=0, atol=0)

        # A nonaligned prefix moves the injection at a former chunk boundary
        # into another chunk, so a mistaken carry or tail changes continuation.
        split = min(65, length - 1)
        prefix_inputs = {
            name: value[:, :, :split].contiguous() for name, value in inputs.items()
        }
        suffix_inputs = {
            name: value[:, :, split:].contiguous() for name, value in inputs.items()
        }
        with torch.no_grad():
            prefix = compiled(
                **prefix_inputs,
                initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
            suffix = compiled(
                **suffix_inputs,
                initial_state=prefix.final_state,
                return_final_state=True,
            )
        continued = torch.cat(
            (prefix.outputs["output"], suffix.outputs["output"]), dim=2
        )
        torch.testing.assert_close(continued.cpu(), expected, rtol=0.02, atol=0.05)
        assert suffix.final_state is not None
        torch.testing.assert_close(
            suffix.final_state["memory"].cpu(), expected_final, rtol=0.01, atol=0.02
        )

        # Kernel compilation happens above, outside capture. Fresh output and
        # scratch allocations during capture must remain valid for replay.
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            captured = compiled(
                **inputs,
                initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
        graph.replay()
        _assert_result(captured, expected, expected_final)


@pytest.mark.parametrize("length", [65, 130, 2048])
def test_scalar_signed_zero_gates_and_grouped_heads(length, gpu_device):
    """Zero factors cannot be handled by an inverse gate or log(prefix) ratio."""
    heads, key_heads, value_heads, rows, columns = 4, 2, 1, 128, 64
    compiled = _compile(
        heads=heads,
        key_heads=key_heads,
        value_heads=value_heads,
        rows=rows,
        columns=columns,
        dtype=torch.float32,
        gate_kind="direct",
    )
    inputs, initial = _inputs(
        2,
        heads,
        key_heads,
        value_heads,
        length,
        rows,
        columns,
        torch.float32,
        "direct",
        gpu_device,
    )
    expected, expected_final = _reference(inputs, initial, "direct")
    with torch.no_grad():
        result = compiled(
            **inputs,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    _assert_result(result, expected, expected_final)


def test_log_gate_underflow_does_not_destroy_later_injections(gpu_device):
    length, heads, rows, columns = 130, 2, 64, 128
    compiled = _compile(
        heads=heads,
        key_heads=heads,
        value_heads=heads,
        rows=rows,
        columns=columns,
        dtype=torch.float32,
        gate_kind="log",
    )
    inputs, initial = _inputs(
        1,
        heads,
        heads,
        heads,
        length,
        rows,
        columns,
        torch.float32,
        "log",
        gpu_device,
    )
    inputs["gate"][:, :, 63:65] = -1000.0
    expected, expected_final = _reference(inputs, initial, "log")
    with torch.no_grad():
        result = compiled(
            **inputs,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    _assert_result(result, expected, expected_final)


def test_one_scalar_executable_handles_changing_batch_occupancy(gpu_device):
    heads, rows, columns, length = 2, 64, 64, 130
    compiled = _compile(
        heads=heads,
        key_heads=heads,
        value_heads=heads,
        rows=rows,
        columns=columns,
        dtype=torch.float32,
        gate_kind="log",
    )
    for batch in (1, 16, 2):
        inputs, initial = _inputs(
            batch,
            heads,
            heads,
            heads,
            length,
            rows,
            columns,
            torch.float32,
            "log",
            gpu_device,
        )
        expected, expected_final = _reference(inputs, initial, "log")
        with torch.no_grad():
            result = compiled(
                **inputs,
                initial_state=StateTuple(("memory",), (initial,)),
                return_final_state=True,
            )
        _assert_result(result, expected, expected_final)


@pytest.mark.parametrize("gate_kind", [None, "direct", "log"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_raw_scalar_parallel_row_column_tails_and_continuation(
    gate_kind, dtype, gpu_device
):
    """Exercise padded GEMM tiles even for a short continuation suffix."""
    from attn_engine.stateful.lowering.scalar.scalar import (
        tilelang_scalar_factorized_dense_raw,
    )

    heads, rows, columns, length = 4, 35, 67, 130
    inputs, initial = _inputs(
        1, heads, 2, 1, length, rows, columns, dtype, gate_kind, gpu_device
    )

    def binding(name, scale=1.0, log=False):
        return (
            SimpleNamespace(name=name),
            SimpleNamespace(operation="identity", scale=scale, is_log_scale=log),
        )

    recipe = SimpleNamespace(
        propagation=binding(
            "__identity_propagation__" if gate_kind is None else "gate",
            log=gate_kind == "log",
        ),
        left_factor=binding("key"),
        right_factor=binding("value"),
        readout=binding("query", rows**-0.5),
    )

    def execute(values, state, write_final):
        return tilelang_scalar_factorized_dense_raw(
            values,
            state,
            recipe,
            sequence_count=1,
            sequence_length=values["query"].shape[2],
            output_dtype=dtype,
            return_final_state=write_final,
        )

    expected, expected_final = _reference(inputs, initial, gate_kind)
    before = initial.clone()
    with torch.no_grad():
        output, final = execute(inputs, initial, True)
        output_only, absent = execute(inputs, initial, False)
        prefix_inputs = {
            name: value[:, :, :65].contiguous() for name, value in inputs.items()
        }
        suffix_inputs = {
            name: value[:, :, 65:].contiguous() for name, value in inputs.items()
        }
        prefix, prefix_final = execute(prefix_inputs, initial, True)
        suffix, suffix_final = execute(suffix_inputs, prefix_final, True)
    assert absent is None
    assert final.dtype == torch.float32 and suffix_final.dtype == torch.float32
    for result in (output, output_only, torch.cat((prefix, suffix), dim=2)):
        torch.testing.assert_close(result.cpu(), expected, rtol=0.02, atol=0.05)
    for state in (final, suffix_final):
        torch.testing.assert_close(state.cpu(), expected_final, rtol=0.01, atol=0.02)
    torch.testing.assert_close(initial, before, rtol=0, atol=0)


@pytest.mark.parametrize("length", [65, 130])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate_kind", ["direct", "log"])
def test_nested_scales_public_scalar_execution(length, dtype, gate_kind, gpu_device):
    from dataclasses import replace

    original = _compile(
        heads=2,
        key_heads=1,
        value_heads=1,
        rows=64,
        columns=64,
        dtype=dtype,
        gate_kind=gate_kind,
    )
    base = original.operator.algorithm
    transition = base.transition
    gate = (Input("gate") * 0.5) * 0.5
    factors = transition.injections[0].factors
    algorithm = AlgorithmIR(
        base.inputs,
        base.states,
        replace(
            transition,
            propagations=(AxisScale(Exp(gate) if gate_kind == "log" else gate),),
            injections=(
                ProductInjection(
                    (
                        ProductFactor((Input("key") * 0.1) * 0.3, factors[0].roles),
                        ProductFactor((Input("value") * 0.3) * 0.1, factors[1].roles),
                    )
                ),
            ),
        ),
        (replace(base.readouts[0], operand=(Input("query") * 0.5) * 0.5),),
        base.head_mapping,
    )
    operator = StatefulOperator(algorithm)
    compiled = operator.compile(
        role_sizes={
            QUERY_HEADS: 2,
            KEY_HEADS: 1,
            VALUE_HEADS: 1,
            STATE_HEADS: 2,
            KEY: 64,
            VALUE: 64,
        },
        target=Target(),
        mode="dense",
        optimized=True,
    )
    assert operator._plans[compiled.specialization].plan.candidate.lowerer == (
        "tilelang-scalar-factorized"
    )
    inputs, initial = _inputs(
        1,
        2,
        1,
        1,
        length,
        64,
        64,
        dtype,
        gate_kind,
        gpu_device,
    )
    # Evaluate each authored multiplication separately with rounded FP32 literals.
    query = (inputs["query"].cpu().float() * 0.5) * 0.5
    key = (inputs["key"].cpu().float() * torch.tensor(0.1)) * torch.tensor(0.3)
    value = (inputs["value"].cpu().float() * torch.tensor(0.3)) * torch.tensor(0.1)
    key = key.repeat_interleave(2, dim=1)
    value = value.repeat_interleave(2, dim=1)
    gate = (inputs["gate"].cpu().float() * 0.5) * 0.5
    if gate_kind == "log":
        gate = gate.exp()
    gate = gate.repeat_interleave(2, dim=1)
    state = initial.cpu().clone()
    outputs = []
    for token in range(length):
        state = gate[:, :, token, None, None] * state
        state = state + key[:, :, token, :, None] * value[:, :, token, None, :]
        outputs.append(torch.einsum("bhr,bhrc->bhc", query[:, :, token], state))
    expected = torch.stack(outputs, dim=2).to(dtype)
    with torch.no_grad():
        result = compiled(
            **inputs,
            initial_state=StateTuple(("memory",), (initial,)),
            return_final_state=True,
        )
    _assert_result(result, expected, state)


@pytest.mark.parametrize("scales", [(0.1, 0.3), (1e10, 1e-10)])
def test_nested_scale_transform_preserves_fp32_rounding(scales, gpu_device):
    import tilelang
    import tilelang.language as T
    from attn_engine import Constant
    from attn_engine.stateful.lowering.scalar.scalar_kernels import _transform

    steps = tuple(Constant(scale).value for scale in scales)
    transform = _transform(T, ("identity", 1.0, False, 0.0, steps))

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Input: T.Tensor((128,), "float32"), Output: T.Tensor((128,), "float32")
        ):
            with T.Kernel(1, threads=128):
                for index in T.Parallel(128):
                    Output[index] = transform(Input[index])

        return kernel

    generator = torch.Generator().manual_seed(0)
    values = torch.randn(128, generator=generator, dtype=torch.float32)
    if scales[0] == 1e10:
        values *= 1e30
    expected = values.clone()
    for step in steps:
        expected = expected * step
    folded = values * torch.tensor(steps[0] * steps[1])
    assert not torch.equal(expected, folded)
    output = torch.empty_like(values, device=gpu_device)
    build_kernel()(values.to(gpu_device), output)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [65, 130])
def test_nested_scale_public_rounding_and_cache_identity(length, gpu_device):
    from dataclasses import replace
    from attn_engine import Constant

    original = _compile(
        heads=1,
        key_heads=1,
        value_heads=1,
        rows=64,
        columns=64,
        dtype=torch.float32,
        gate_kind=None,
    )
    base = original.operator.algorithm
    values = torch.randn(length, generator=torch.Generator().manual_seed(0))
    inputs = {
        name: torch.zeros(1, 1, length, 64, device=gpu_device)
        for name in ("query", "key", "value")
    }
    inputs["query"][0, 0, :, 0] = values.to(gpu_device)
    initial = torch.zeros(1, 1, 64, 64, device=gpu_device)
    initial[0, 0, 0, 0] = 1.0
    role_sizes = {
        QUERY_HEADS: 1,
        KEY_HEADS: 1,
        VALUE_HEADS: 1,
        STATE_HEADS: 1,
        KEY: 64,
        VALUE: 64,
    }
    # The contraction uses the existing TensorFloat GEMM approximation. Compare
    # with that same contraction over independently precomputed authored scales,
    # rather than expecting the contraction itself to be a bitwise identity.
    reference_operator = StatefulOperator(
        AlgorithmIR(
            base.inputs,
            base.states,
            base.transition,
            (replace(base.readouts[0], operand=Input("query")),),
            base.head_mapping,
        )
    )
    reference = reference_operator.compile(
        role_sizes=role_sizes,
        target=Target(),
        mode="dense",
        optimized=True,
    )
    results = []
    for scales in ((0.1, 0.3), (0.3, 0.2)):
        algorithm = AlgorithmIR(
            base.inputs,
            base.states,
            base.transition,
            (
                replace(
                    base.readouts[0], operand=(Input("query") * scales[0]) * scales[1]
                ),
            ),
            base.head_mapping,
        )
        operator = StatefulOperator(algorithm)
        compiled = operator.compile(
            role_sizes={
                QUERY_HEADS: 1,
                KEY_HEADS: 1,
                VALUE_HEADS: 1,
                STATE_HEADS: 1,
                KEY: 64,
                VALUE: 64,
            },
            target=Target(),
            mode="dense",
            optimized=True,
        )
        assert operator._plans[compiled.specialization].plan.candidate.lowerer == (
            "tilelang-scalar-factorized"
        )
        with torch.no_grad():
            result = (
                compiled(**inputs, initial_state=StateTuple(("memory",), (initial,)))
                .outputs["output"][0, 0, :, 0]
                .cpu()
            )
        scaled = (values * Constant(scales[0]).value) * Constant(scales[1]).value
        reference_inputs = dict(inputs)
        reference_inputs["query"] = torch.zeros_like(inputs["query"])
        reference_inputs["query"][0, 0, :, 0] = scaled.to(gpu_device)
        with torch.no_grad():
            expected = (
                reference(
                    **reference_inputs,
                    initial_state=StateTuple(("memory",), (initial,)),
                )
                .outputs["output"][0, 0, :, 0]
                .cpu()
            )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        results.append(result)
    assert not torch.equal(*results)
