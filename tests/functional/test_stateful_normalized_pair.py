"""Normalized identity States preserve independent carries through public calls."""

from __future__ import annotations

import pytest
import torch

from attn_engine import (
    AlgorithmIR,
    Batch,
    Constant,
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
    StatefulOperator,
    StateTuple,
    Target,
    TensorInput,
)
from examples.normalized_linear_attention import (
    EPS,
    KEY,
    KEY_HEADS,
    QUERY_HEADS,
    STATE_HEADS,
    VALUE,
    VALUE_HEADS,
)

pytestmark = [pytest.mark.functional, pytest.mark.gpu]
NORMALIZER_HEADS = HeadRole("normalizer_heads")


def _operator(dtype, layout, dim, prior_memory, prior_normalizer):
    def spec(name, head, feature):
        roles = (
            (Batch, Sequence, head, feature)
            if layout == "bth"
            else (Batch, head, Sequence, feature)
        )
        return TensorInput(name, roles, dtype)

    scaled_query = Input("query") * dim**-0.5
    return StatefulOperator(
        AlgorithmIR(
            inputs=(
                spec("query", QUERY_HEADS, KEY),
                spec("key", KEY_HEADS, KEY),
                spec("value", VALUE_HEADS, VALUE),
                spec("denominator_key", NORMALIZER_HEADS, KEY),
            ),
            states=(
                StateSpec("memory", (Batch, STATE_HEADS, KEY, VALUE)),
                StateSpec("key_sum", (Batch, STATE_HEADS, KEY)),
            ),
            transition=(
                StateTransition(
                    "memory",
                    injections=(
                        ProductInjection(
                            (
                                ProductFactor(Input("key"), (KEY,)),
                                ProductFactor(Input("value"), (VALUE,)),
                            )
                        ),
                    ),
                ),
                StateTransition(
                    "key_sum",
                    injections=(
                        ProductInjection(
                            (ProductFactor(Input("denominator_key"), (KEY,)),)
                        ),
                    ),
                ),
            ),
            readouts=(
                Readout(
                    "output",
                    StateTerm("memory", scaled_query, KEY, prior=prior_memory)
                    / (
                        StateTerm("key_sum", scaled_query, KEY, prior=prior_normalizer)
                        + Constant(EPS)
                    ),
                    dtype,
                ),
            ),
            head_mapping=HeadMapping(
                {
                    role: STATE_HEADS
                    for role in (
                        QUERY_HEADS,
                        KEY_HEADS,
                        VALUE_HEADS,
                        NORMALIZER_HEADS,
                        STATE_HEADS,
                    )
                }
            ),
        )
    )


def _compile(operator, heads, dim, dim_value, source_heads):
    qh, kh, vh, nh = source_heads
    return operator.compile(
        role_sizes={
            QUERY_HEADS: qh,
            KEY_HEADS: kh,
            VALUE_HEADS: vh,
            NORMALIZER_HEADS: nh,
            STATE_HEADS: heads,
            KEY: dim,
            VALUE: dim_value,
        },
        target=Target(),
        mode="dense",
        optimized=True,
    )


def _snapshot(state):
    return {name: state[name].clone() for name in state.names}


def _assert_initial(before, state, result):
    for name in state.names:
        tensor = state[name]
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        if result.final_state is not None:
            for final in result.final_state:
                assert (
                    final.untyped_storage().data_ptr()
                    != tensor.untyped_storage().data_ptr()
                )


def _reference(inputs, memory, key_sum, prior_memory, prior_normalizer):
    """CPU FP32 recurrence; only authored query arithmetic rounds to input dtype."""
    heads = memory.shape[1]
    expanded = {
        name: tensor.repeat_interleave(heads // tensor.shape[1], dim=1)
        for name, tensor in inputs.items()
    }
    query, key, value, denominator_key = (
        expanded[name] for name in ("query", "key", "value", "denominator_key")
    )
    scale = torch.tensor(query.shape[-1] ** -0.5, dtype=query.dtype)
    memory, key_sum = memory.clone(), key_sum.clone()
    outputs = []
    for token in range(query.shape[2]):
        next_memory = memory + key[:, :, token].float().unsqueeze(-1) * value[
            :, :, token
        ].float().unsqueeze(-2)
        next_key_sum = key_sum + denominator_key[:, :, token].float()
        scaled = (query[:, :, token] * scale).float()
        numerator = (
            scaled.unsqueeze(-1) * (memory if prior_memory else next_memory)
        ).sum(-2)
        denominator = (scaled * (key_sum if prior_normalizer else next_key_sum)).sum(
            -1, keepdim=True
        )
        outputs.append(numerator / (denominator + EPS))
        memory, key_sum = next_memory, next_key_sum
    return torch.stack(outputs, 2).to(query.dtype).float(), memory, key_sum


@pytest.mark.parametrize(
    "prior_memory,prior_normalizer",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length,layout", [(1, "bht"), (64, "bth"), (71, "bth")])
def test_normalized_independent_keys_prior_and_continuation(
    gpu_device,
    dtype,
    length,
    layout,
    prior_memory,
    prior_normalizer,
):
    batch, heads, dim, dim_value = 2, 4, 17, 35
    source_heads = (2, 1, 2, 4)
    generator = torch.Generator().manual_seed(367)
    host = {
        name: (
            torch.rand(batch, head, length, feature, generator=generator) + offset
        ).to(dtype)
        for name, head, feature, offset in (
            ("query", 2, dim, 0.1),
            ("key", 1, dim, 0.1),
            ("value", 2, dim_value, -0.5),
            ("denominator_key", 4, dim, 0.5),
        )
    }
    memory = torch.rand(batch, heads, dim, dim_value, generator=generator) * 0.1 + 0.1
    key_sum = torch.rand(batch, heads, dim, generator=generator) + 0.5
    expected, final_memory, final_key_sum = _reference(
        host,
        memory,
        key_sum,
        prior_memory,
        prior_normalizer,
    )
    operator = _operator(dtype, layout, dim, prior_memory, prior_normalizer)
    compiled = _compile(operator, heads, dim, dim_value, source_heads)
    inputs = {
        name: (tensor.transpose(1, 2).contiguous() if layout == "bth" else tensor).to(
            gpu_device
        )
        for name, tensor in host.items()
    }
    initial = StateTuple(
        ("memory", "key_sum"), (memory.to(gpu_device), key_sum.to(gpu_device))
    )
    before = _snapshot(initial)
    full = compiled(**inputs, initial_state=initial, return_final_state=True)
    _assert_initial(before, initial, full)
    output_only = compiled(**inputs, initial_state=initial)
    assert output_only.final_state is None
    _assert_initial(before, initial, output_only)
    outputs, finals = [full.outputs["output"], output_only.outputs["output"]], [full]
    if length == 71:
        axis, split = (1 if layout == "bth" else 2), 65
        prefix = compiled(
            **{
                name: tensor.narrow(axis, 0, split).contiguous()
                for name, tensor in inputs.items()
            },
            initial_state=initial,
            return_final_state=True,
        )
        _assert_initial(before, initial, prefix)
        prefix_before = _snapshot(prefix.final_state)
        suffix = compiled(
            **{
                name: tensor.narrow(axis, split, length - split).contiguous()
                for name, tensor in inputs.items()
            },
            initial_state=prefix.final_state,
            return_final_state=True,
        )
        _assert_initial(before, initial, suffix)
        _assert_initial(prefix_before, prefix.final_state, suffix)
        outputs.append(
            torch.cat((prefix.outputs["output"], suffix.outputs["output"]), 2)
        )
        finals.append(suffix)
    for actual in outputs:
        torch.testing.assert_close(
            actual.float().cpu(),
            expected,
            rtol=1e-2 if dtype == torch.bfloat16 else 2e-4,
            atol=2e-5,
        )
    for result in finals:
        assert result.final_state.names == ("memory", "key_sum")
        for name, expected_state in (
            ("memory", final_memory),
            ("key_sum", final_key_sum),
        ):
            assert result.final_state[name].dtype == torch.float32
            torch.testing.assert_close(
                result.final_state[name].cpu(), expected_state, rtol=2e-4, atol=2e-5
            )
    if dtype == torch.bfloat16:
        # Lossless promotion cannot alter FP32 injection or final-State accumulation.
        promoted = _compile(
            _operator(torch.float32, layout, dim, prior_memory, prior_normalizer),
            heads,
            dim,
            dim_value,
            source_heads,
        )
        promoted_result = promoted(
            **{name: tensor.float() for name, tensor in inputs.items()},
            initial_state=initial,
            return_final_state=True,
        )
        _assert_initial(before, initial, promoted_result)
        for name in full.final_state.names:
            torch.testing.assert_close(
                full.final_state[name],
                promoted_result.final_state[name],
                rtol=0,
                atol=0,
            )


def test_normalized_hierarchical_exclusive_prefix_boundary(gpu_device):
    """65 chunks: exact integer prefixes expose overwritten summaries and group carry."""
    length, dim, dim_value = 4097, 4, 3
    token = torch.arange(length)
    key = (2.0 ** (token % 3))[:, None] * torch.tensor([1.0, 2.0, 1.0, 2.0])
    value = (2.0 ** ((token // 7) % 2))[:, None] * torch.tensor([1.0, 2.0, 4.0])
    denominator_key = (2.0 ** ((token // 11) % 2))[:, None] * torch.tensor(
        [2.0, 1.0, 4.0, 2.0]
    )
    query = torch.tensor([1.0, 2.0, 1.0, 2.0]).expand(length, dim)
    memory = torch.tensor([1.0, 2.0, 4.0]).expand(1, 1, dim, dim_value).clone()
    key_sum = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(1, 1, dim)
    # All products, prefixes and contractions are exactly representable in FP32.
    injections = key.unsqueeze(-1) * value.unsqueeze(-2)
    memory_prefix = memory[0, 0] + injections.cumsum(0)
    normalizer_prefix = key_sum[0, 0] + denominator_key.cumsum(0)
    prior_memory = torch.cat((memory[0, 0].unsqueeze(0), memory_prefix[:-1]))
    prior_normalizer = torch.cat((key_sum[0, 0].unsqueeze(0), normalizer_prefix[:-1]))
    scaled = query * 0.5
    expected = (
        (scaled.unsqueeze(-1) * prior_memory).sum(-2)
        / ((scaled * prior_normalizer).sum(-1, keepdim=True) + EPS)
    ).reshape(1, 1, length, dim_value)
    inputs = {
        name: tensor.reshape(1, 1, length, -1).to(gpu_device)
        for name, tensor in (
            ("query", query),
            ("key", key),
            ("value", value),
            ("denominator_key", denominator_key),
        )
    }
    compiled = _compile(
        _operator(torch.float32, "bht", dim, True, True),
        1,
        dim,
        dim_value,
        (1, 1, 1, 1),
    )
    initial = StateTuple(
        ("memory", "key_sum"), (memory.to(gpu_device), key_sum.to(gpu_device))
    )
    before = _snapshot(initial)
    full = compiled(**inputs, initial_state=initial, return_final_state=True)
    _assert_initial(before, initial, full)
    output_only = compiled(**inputs, initial_state=initial)
    assert output_only.final_state is None
    _assert_initial(before, initial, output_only)
    prefix = compiled(
        **{name: tensor[:, :, :4096].contiguous() for name, tensor in inputs.items()},
        initial_state=initial,
        return_final_state=True,
    )
    _assert_initial(before, initial, prefix)
    prefix_before = _snapshot(prefix.final_state)
    suffix = compiled(
        **{name: tensor[:, :, 4096:].contiguous() for name, tensor in inputs.items()},
        initial_state=prefix.final_state,
        return_final_state=True,
    )
    _assert_initial(before, initial, suffix)
    _assert_initial(prefix_before, prefix.final_state, suffix)
    for actual in (
        full.outputs["output"],
        output_only.outputs["output"],
        torch.cat((prefix.outputs["output"], suffix.outputs["output"]), 2),
    ):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-5)
    for result, last in ((prefix, 4095), (full, 4096), (suffix, 4096)):
        assert result.final_state.names == ("memory", "key_sum")
        torch.testing.assert_close(
            result.final_state["memory"].cpu(),
            memory_prefix[last].reshape_as(memory),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.final_state["key_sum"].cpu(),
            normalizer_prefix[last].reshape_as(key_sum),
            rtol=0,
            atol=0,
        )
