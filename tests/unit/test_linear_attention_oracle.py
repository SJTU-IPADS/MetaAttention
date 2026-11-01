"""The independent recurrence must preserve authored BF16 expression nodes."""

import pytest
import torch

from tests.functional.test_linear_attention_examples import _reference
from tests.functional.test_two_axis_elementwise import _reference as _two_axis_reference

pytestmark = pytest.mark.unit


def test_two_axis_reference_preserves_authored_key_and_query_expressions():
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 1, 128, generator=generator).to(torch.bfloat16)
    key = torch.full_like(query, 0.3)
    value = torch.full_like(query, 0.2)
    initial = torch.zeros(1, 1, 128, 128)
    output, state = _two_axis_reference(
        {
            "query": query,
            "key": key,
            "value": value,
            "key_gate": torch.zeros_like(key, dtype=torch.float32),
            "value_gate": torch.zeros_like(value, dtype=torch.float32),
        },
        (1,),
        initial,
        128**-0.5,
    )
    expected_state = (key + 0.1).float()[..., None] * value.float()[..., None, :]
    expected_output = torch.einsum(
        "thk,thkv->thv", (query * 128**-0.5).float(), expected_state
    ).to(query.dtype)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


def test_reference_rounds_key_expression_before_fp32_state_update():
    query = torch.full((1, 1, 1, 64), 0.7, dtype=torch.bfloat16)
    key = torch.full_like(query, 0.3)
    value = torch.full_like(query, 0.2)
    _, state = _reference(query, key, value)
    expected = (key + 0.1).float()[:, :, 0, :, None] * value.float()[:, :, 0, None, :]
    torch.testing.assert_close(state, expected, rtol=0, atol=0)


def test_reference_rounds_query_expression_before_fp32_contraction():
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 1, 1, 128, generator=generator).to(torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    initial = torch.randn(1, 1, 128, 128, generator=generator)
    actual, _ = _reference(query, key, value, initial_state=initial)
    expected = ((query * 128**-0.5).float()[:, :, 0, :, None] * initial).sum(-2)
    torch.testing.assert_close(
        actual, expected.unsqueeze(2).to(query.dtype), rtol=0, atol=0
    )
