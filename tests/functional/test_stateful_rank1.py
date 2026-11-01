"""Public vector scans preserve tails, zero gates, head sharing and continuation."""

from __future__ import annotations

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Constant,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    Readout,
    Sequence,
    StateSpec,
    StateTerm,
    StateTransition,
    StatefulCompilationError,
    StatefulOperator,
    StateTuple,
    Target,
    TensorInput,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize("length", [0, 1, 63, 64, 65, 130, 2048, 4097, 8192])
@pytest.mark.parametrize("mode", ["log", "signed", "identity"])
def test_vector_state_tail_and_continuation(gpu_device, length, mode):
    feature = FeatureRole("vector")
    source = HeadRole("source")
    target = HeadRole("target")
    dtype = torch.float32 if mode == "signed" else torch.bfloat16
    inputs = [TensorInput("value", (Batch, source, Sequence, feature), dtype)]
    propagations = ()
    if mode != "identity":
        inputs.append(
            TensorInput("gate", (Batch, target, Sequence, feature), torch.float32)
        )
        expression = Exp(Input("gate")) if mode == "log" else Input("gate")
        propagations = (AxisScale(expression, (feature,)),)
    algorithm = AlgorithmIR(
        inputs=tuple(inputs),
        states=(StateSpec("hidden", (Batch, target, feature)),),
        transition=StateTransition(
            "hidden",
            propagations=propagations,
            injections=(
                ProductInjection((ProductFactor(Input("value"), (feature,)),)),
            ),
        ),
        readouts=(Readout("output", StateTerm("hidden", Constant(1.0), None), dtype),),
        head_mapping=HeadMapping({source: target, target: target}),
    )
    compiled = StatefulOperator(algorithm).compile(
        role_sizes={source: 2, target: 8, feature: 37}, target=Target(), mode="dense"
    )
    generator = torch.Generator().manual_seed(939)
    value = (torch.randn(2, 2, length, 37, generator=generator) * 0.1).to(dtype)
    initial = torch.randn(2, 8, 37, generator=generator)
    gate = -torch.rand(2, 8, length, 37, generator=generator)
    if mode == "signed":
        gate = gate * 1.5 + 0.5
    if length > 64 and mode != "identity":
        gate[:, :, 63] = float("-inf") if mode == "log" else 0
        gate[:, :, 64] = 0 if mode == "log" else -0.5
    values = {"value": value.to(gpu_device)}
    if mode != "identity":
        values["gate"] = gate.to(gpu_device)
    initial_state = StateTuple(("hidden",), (initial.to(gpu_device),))
    expected = []
    state = initial.clone()
    expanded = value.float().repeat_interleave(4, dim=1)
    for token in range(length):
        if mode != "identity":
            factor = gate[:, :, token].exp() if mode == "log" else gate[:, :, token]
            state = state * factor
        state = state + expanded[:, :, token]
        expected.append(state.clone())
    expected = torch.stack(expected, 2) if expected else torch.empty(2, 8, 0, 37)
    if length == 0:
        with pytest.raises(StatefulCompilationError) as error:
            compiled(**values, initial_state=initial_state, return_final_state=True)
        assert error.value.category == "RUNTIME_SHAPE"
        return
    result = compiled(**values, initial_state=initial_state, return_final_state=True)
    torch.testing.assert_close(
        result.outputs["output"].cpu().float(),
        expected.to(dtype).float(),
        rtol=2e-2 if dtype == torch.bfloat16 else 2e-5,
        atol=2e-3 if dtype == torch.bfloat16 else 2e-5,
    )
    torch.testing.assert_close(
        result.final_state["hidden"].cpu(), state, rtol=2e-5, atol=2e-5
    )
    output_only = compiled(**values, initial_state=initial_state)
    assert output_only.final_state is None
    torch.testing.assert_close(output_only.outputs["output"], result.outputs["output"])
    torch.testing.assert_close(initial_state["hidden"].cpu(), initial, rtol=0, atol=0)
    if length == 1:
        return
    split = length // 2
    first = compiled(
        **{name: tensor[:, :, :split].contiguous() for name, tensor in values.items()},
        initial_state=initial_state,
        return_final_state=True,
    )
    second = compiled(
        **{name: tensor[:, :, split:].contiguous() for name, tensor in values.items()},
        initial_state=first.final_state,
        return_final_state=True,
    )
    actual = torch.cat((first.outputs["output"], second.outputs["output"]), 2)
    torch.testing.assert_close(
        actual.cpu().float(),
        expected.to(dtype).float(),
        rtol=2e-2 if dtype == torch.bfloat16 else 2e-5,
        atol=2e-3 if dtype == torch.bfloat16 else 2e-5,
    )
    torch.testing.assert_close(
        second.final_state["hidden"].cpu(), state, rtol=2e-5, atol=2e-5
    )


def test_vector_autotune_preserves_rank_one_state(gpu_device):
    from examples.hgrn import hgrn
    from testing.benchmark_stateful_plans import CASES, role_sizes_for
    from dataclasses import replace

    case = replace(
        next(case for case in CASES if case.name == "hgrn"), heads=1, dim_value=11
    )
    operator = hgrn(1, 1, 65, 11, dtype=torch.float32)
    compiled = operator.compile(
        role_sizes=role_sizes_for(operator, case, 65),
        target=Target(),
        mode="dense",
        autotune=True,
    )
    generator = torch.Generator().manual_seed(947)
    value = torch.randn(1, 1, 65, 11, generator=generator) * 0.1
    gate = -torch.rand(1, 1, 65, 11, generator=generator)
    initial = torch.randn(1, 1, 11, generator=generator) * 0.1
    state, expected = initial.clone(), []
    for token in range(65):
        state = state * gate[:, :, token].exp() + value[:, :, token]
        expected.append(state.clone())
    result = compiled(
        value=value.to(gpu_device),
        gate=gate.to(gpu_device),
        initial_state=StateTuple(("hidden",), (initial.to(gpu_device),)),
        return_final_state=True,
    )
    torch.testing.assert_close(
        result.outputs["output"].cpu(), torch.stack(expected, 2), rtol=2e-5, atol=2e-5
    )
    torch.testing.assert_close(
        result.final_state["hidden"].cpu(), state, rtol=2e-5, atol=2e-5
    )


def test_one_compiled_vector_rebinds_dynamic_workloads_and_storage(gpu_device):
    feature, source, heads = (
        FeatureRole("vector"),
        HeadRole("source"),
        HeadRole("heads"),
    )
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=(
                TensorInput("value", (Batch, Sequence, source, feature), torch.float32),
                TensorInput("gate", (Batch, Sequence, feature), torch.float32),
            ),
            states=(StateSpec("hidden", (Batch, heads, feature)),),
            transition=StateTransition(
                "hidden",
                propagations=(AxisScale(Input("gate"), (feature,)),),
                injections=(
                    ProductInjection((ProductFactor(Input("value"), (feature,)),)),
                ),
            ),
            readouts=(
                Readout(
                    "output", StateTerm("hidden", Constant(1.0), None), torch.float32
                ),
            ),
            head_mapping=HeadMapping({source: heads}),
        )
    )
    compiled = operator.compile(
        role_sizes={source: 2, heads: 4, feature: 7}, target=Target(), mode="dense"
    )
    generator = torch.Generator().manual_seed(953)
    retained = []
    for batch, length in ((1, 1), (2, 65), (1, 130), (3, 17)):
        value = torch.randn(batch, length, 2, 7, generator=generator) * 0.1
        gate = torch.rand(batch, length, 7, generator=generator) * 1.5 - 0.5
        initial = torch.randn(batch, 4, 7, generator=generator) * 0.1
        # Fresh contiguous tensors with nonzero offsets: a recipe must never
        # retain old storage or compute strides from the previous B/T shape.
        device_inputs = {}
        for name, host in (("value", value), ("gate", gate)):
            storage = torch.empty(host.numel() + 19, device=gpu_device)
            tensor = storage[19:].reshape(host.shape)
            tensor.copy_(host)
            device_inputs[name] = tensor
        state, expected = initial.clone(), []
        for token in range(length):
            state = state * gate[:, token, None, :] + value[:, token].repeat_interleave(
                2, dim=1
            )
            expected.append(state.clone())
        result = compiled(
            **device_inputs,
            initial_state=StateTuple(("hidden",), (initial.to(gpu_device),)),
            return_final_state=True,
        )
        expected = torch.stack(expected, 2)
        torch.testing.assert_close(
            result.outputs["output"].cpu(), expected, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(
            result.final_state["hidden"].cpu(), state, rtol=2e-5, atol=2e-5
        )
        retained.append((result.outputs["output"], expected))
        invalid = dict(device_inputs, gate=device_inputs["gate"].to(torch.bfloat16))
        with pytest.raises(StatefulCompilationError) as error:
            compiled(**invalid)
        assert error.value.category == "RUNTIME_DTYPE"
    for actual, expected in retained:
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-5)
