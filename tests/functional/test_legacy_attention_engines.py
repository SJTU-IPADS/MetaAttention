from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from attn_engine import StateTuple, Target
from examples.gated_retention import gated_retention
from examples.mamba2 import mamba2
from examples.mha import causal_softmax_attention
from examples.mha_decode import softmax_attention_decode
from examples.mla_decode import mla_decode
from examples.reluattn import relu_attention
from examples.retnet_recurrent import retnet_recurrent
from examples.sigmoid_attn import sigmoid_attention
from examples.sparse_gqa_decode import sparse_gqa_decode
from benchmark.bench_utils import assert_close

from .reference import (
    causal_softmax,
    decode_softmax,
    gated_retention as gated_retention_reference,
    mla_decode as mla_decode_reference,
    mamba2 as mamba2_reference,
    relu_attention as relu_reference,
    retnet_recurrent as retnet_reference,
    sigmoid_attention as sigmoid_reference,
    sparse_gqa_decode as sparse_reference,
)


def _compile_stateful(
    operator,
    *,
    query_heads,
    key_heads,
    value_heads,
    state_heads,
    key_dim,
    value_dim,
    mode="dense",
    backward=False,
):
    roles = {
        role.name: role
        for spec in (*operator.algorithm.inputs, *operator.algorithm.states)
        for role in spec.roles
        if hasattr(role, "name")
    }
    return operator.compile(
        role_sizes={
            roles["query_heads"]: query_heads,
            roles["key_heads"]: key_heads,
            roles["value_heads"]: value_heads,
            roles["state_heads"]: state_heads,
            roles["key_dim"]: key_dim,
            roles["value_dim"]: value_dim,
        },
        target=Target(),
        mode=mode,
        backward=backward,
    )


pytestmark = [pytest.mark.functional, pytest.mark.gpu, pytest.mark.slow]


SOFTMAX_PREFILL_CASES = [
    pytest.param(1, 16, 2048, 128, 128, id="B1-H16-S2048-D128-DV128"),
    pytest.param(1, 16, 2048, 128, 256, id="B1-H16-S2048-D128-DV256"),
]


def _backward_compare(actual, expected, upstream, pairs, **kwargs):
    actual.backward(upstream, retain_graph=True)
    expected.backward(upstream, retain_graph=True)
    for actual_tensor, expected_tensor in pairs:
        assert_close(actual_tensor.grad, expected_tensor.grad, **kwargs)


@pytest.mark.parametrize("batch,heads,seqlen,dim,dim_value", SOFTMAX_PREFILL_CASES)
def test_legacy_softmax_prefill(batch, heads, seqlen, dim, dim_value, gpu_device, seed):
    dtype = torch.float16
    module = causal_softmax_attention(batch, heads, seqlen, dim, dim_value, tune=True)
    query = torch.randn(batch, seqlen, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn(batch, seqlen, heads, dim, device=gpu_device, dtype=dtype)
    value = torch.randn(batch, seqlen, heads, dim_value, device=gpu_device, dtype=dtype)
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    query = query.requires_grad_()
    key = key.requires_grad_()
    value = value.requires_grad_()
    actual = module(query, key, value)
    expected = causal_softmax(query_ref, key_ref, value_ref)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    upstream = torch.randn_like(actual)
    _backward_compare(
        actual,
        expected,
        upstream,
        [(query, query_ref), (key, key_ref), (value, value_ref)],
        rtol=1e-1,
        atol=1e-1,
    )


def test_legacy_softmax_decode(gpu_device, seed):
    batch, heads, query_len, kv_len, dim, dim_value = 8, 16, 1, 4096, 128, 128
    dtype = torch.float16
    module = softmax_attention_decode(batch, heads, query_len, kv_len, dim, dim_value)
    query = torch.randn(batch, query_len, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn(batch, kv_len, heads, dim, device=gpu_device, dtype=dtype)
    value = torch.randn(batch, kv_len, heads, dim_value, device=gpu_device, dtype=dtype)
    actual = module(query, key, value)
    expected = decode_softmax(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_legacy_mamba2(gpu_device, seed):
    batch, query_heads, seqlen, dim, dim_value, key_heads, value_heads = (
        1,
        1,
        128,
        128,
        64,
        1,
        4,
    )
    dtype = torch.bfloat16
    module = mamba2(
        batch,
        query_heads,
        seqlen,
        dim,
        dim_value,
        HK=key_heads,
        HV=value_heads,
        dtype=dtype,
    )
    module = _compile_stateful(
        module,
        query_heads=query_heads,
        key_heads=key_heads,
        value_heads=value_heads,
        state_heads=value_heads,
        key_dim=dim,
        value_dim=dim_value,
        backward=True,
    )
    query = torch.randn(batch, seqlen, query_heads, dim, device=gpu_device, dtype=dtype)
    key = 0.5 * torch.randn(
        batch, seqlen, key_heads, dim, device=gpu_device, dtype=dtype
    )
    value = torch.randn(
        batch, seqlen, value_heads, dim_value, device=gpu_device, dtype=dtype
    )
    a_param = 1.5 * torch.rand(value_heads, dtype=dtype, device=gpu_device) - 4
    dt_seed = 0.7 * torch.rand(
        batch, seqlen, value_heads, dtype=torch.float32, device=gpu_device
    )
    dt_min, dt_max = 0.001, 0.1
    dt = torch.exp(
        torch.rand(value_heads, device=gpu_device, dtype=dtype)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp_min(1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    delta_t = F.softplus(dt_seed + dt_bias)
    query_actual = query.transpose(1, 2).contiguous().requires_grad_()
    key_actual = key.transpose(1, 2).contiguous().requires_grad_()
    value_actual = value.transpose(1, 2).contiguous().requires_grad_()
    a_actual = (
        a_param[None, :, None]
        .expand(batch, value_heads, dim)
        .contiguous()
        .requires_grad_()
    )
    delta_actual = delta_t.transpose(1, 2).contiguous().requires_grad_()
    query_ref = query.detach().requires_grad_()
    key_ref = key.detach().requires_grad_()
    value_ref = value.detach().requires_grad_()
    a_ref = a_param.detach().requires_grad_()
    delta_ref = delta_t.detach().requires_grad_()
    actual_result = module(
        query=query_actual,
        key=key_actual,
        value=value_actual,
        A=a_actual,
        dt=delta_actual,
    )
    actual = actual_result.outputs["output"]
    expected = mamba2_reference(value_ref, delta_ref, a_ref, key_ref, query_ref).to(
        dtype
    )
    upstream = 0.1 * torch.randn_like(actual)
    actual.backward(upstream, retain_graph=True)
    expected.backward(upstream.transpose(1, 2), retain_graph=True)
    assert_close(
        actual.transpose(1, 2), expected, rtol=1e-1, atol=1e-1, mismatch_ratio=1e-2
    )
    for actual_tensor, expected_tensor in (
        (query_actual, query_ref),
        (key_actual, key_ref),
        (value_actual, value_ref),
    ):
        assert_close(
            actual_tensor.grad.transpose(1, 2),
            expected_tensor.grad,
            rtol=1e-1,
            atol=1e-1,
            mismatch_ratio=1e-2,
        )


def test_legacy_gated_retention(gpu_device, seed):
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    dtype = torch.bfloat16
    module = gated_retention(batch, heads, seqlen, dim, dim_value, dtype=dtype)
    module = _compile_stateful(
        module,
        query_heads=heads,
        key_heads=heads,
        value_heads=heads,
        state_heads=heads,
        key_dim=dim,
        value_dim=dim_value,
        backward=True,
    )
    query = torch.randn(
        batch, heads, seqlen, dim, device=gpu_device, dtype=dtype
    ).requires_grad_()
    key = torch.randn(
        batch, heads, seqlen, dim, device=gpu_device, dtype=dtype
    ).requires_grad_()
    gate = (
        F.logsigmoid(torch.randn(batch, heads, seqlen, device=gpu_device))
        .clamp_min(-5)
        .requires_grad_()
    )
    value = torch.randn(
        batch, heads, seqlen, dim_value, device=gpu_device, dtype=dtype
    ).requires_grad_()
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    gate_ref = gate.detach().clone().to(dtype).requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    actual = module(query=query, key=key, value=value, gate=gate).outputs["output"]
    expected = gated_retention_reference(query_ref, key_ref, value_ref, gate_ref).to(
        dtype
    )
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
    upstream = torch.randn_like(actual)
    _backward_compare(
        actual,
        expected,
        upstream,
        [(query, query_ref), (key, key_ref), (value, value_ref)],
        rtol=1e-1,
        atol=1e-1,
    )


def test_legacy_gated_retention_unaligned_scalar_carry(gpu_device, seed):
    batch, heads, seqlen, dim, dim_value = 1, 2, 129, 64, 64
    dtype = torch.bfloat16
    module = gated_retention(batch, heads, seqlen, dim, dim_value, dtype=dtype)
    module = _compile_stateful(
        module,
        query_heads=heads,
        key_heads=heads,
        value_heads=heads,
        state_heads=heads,
        key_dim=dim,
        value_dim=dim_value,
    )
    query = torch.randn(batch, heads, seqlen, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    gate = torch.full((batch, heads, seqlen), -0.001, device=gpu_device)
    value = torch.randn(batch, heads, seqlen, dim_value, device=gpu_device, dtype=dtype)
    initial = torch.randn(
        batch, heads, dim, dim_value, device=gpu_device, dtype=torch.float32
    )
    actual = module(
        query=query,
        key=key,
        value=value,
        gate=gate,
        initial_state=StateTuple(("memory",), (initial,)),
    ).outputs["output"]
    query_f, key_f, value_f, gate_f = [x.float() for x in (query, key, value, gate)]
    state = initial.clone()
    expected = torch.empty(batch, heads, seqlen, dim_value, device=gpu_device)
    scale = dim**-0.5
    for token in range(seqlen):
        state = gate_f[:, :, token, None, None].exp() * state
        state = state + key_f[:, :, token, :, None] * value_f[:, :, token, None, :]
        expected[:, :, token] = torch.einsum(
            "bhd,bhdf->bhf", scale * query_f[:, :, token], state
        )
    torch.testing.assert_close(actual, expected.to(dtype), rtol=1e-1, atol=1e-1)


def test_gated_retention_packed_scalar_inference(gpu_device, seed):
    lengths = (0, 65, 64, 0)
    token_count = sum(lengths)
    heads, dim = 2, 64
    dtype = torch.bfloat16
    module = gated_retention(4, heads, 65, dim, dim, dtype=dtype)
    module = _compile_stateful(
        module,
        query_heads=heads,
        key_heads=heads,
        value_heads=heads,
        state_heads=heads,
        key_dim=dim,
        value_dim=dim,
        mode="packed",
    )
    query = torch.randn(token_count, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.full((token_count, heads), -0.01, device=gpu_device)
    offsets = torch.tensor((0, 0, 65, 129, 129), device=gpu_device, dtype=torch.int32)
    initial = torch.randn(4, heads, dim, dim, device=gpu_device, dtype=torch.float32)
    result = module(
        query=query,
        key=key,
        value=value,
        gate=gate,
        sequence_offsets=offsets,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )
    expected = torch.empty_like(query)
    expected_final = initial.clone()
    scale = dim**-0.5
    start = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence]
        for token in range(start, start + length):
            state = gate[token, :, None, None].exp() * state
            state = (
                state
                + key[token, :, :, None].float() * value[token, :, None, :].float()
            )
            expected[token] = torch.einsum(
                "hd,hdf->hf", scale * query[token].float(), state
            ).to(dtype)
        expected_final[sequence] = state
        start += length
    assert result.final_state is not None
    torch.testing.assert_close(result.outputs["output"], expected, rtol=0.1, atol=0.1)
    torch.testing.assert_close(
        result.final_state["memory"], expected_final, rtol=0.1, atol=0.1
    )


@pytest.mark.parametrize("batch", [1, 8])
def test_legacy_sigmoid(batch, gpu_device, seed):
    heads, seqlen, dim, dim_value = 32, 2048, 128, 128
    dtype = torch.float16
    module = sigmoid_attention(batch, heads, seqlen, dim, dim_value, tune=True)
    query = torch.randn(
        batch, seqlen, heads, dim, device=gpu_device, dtype=dtype
    ).requires_grad_()
    key = torch.randn(
        batch, seqlen, heads, dim, device=gpu_device, dtype=dtype
    ).requires_grad_()
    value = torch.randn(
        batch, seqlen, heads, dim_value, device=gpu_device, dtype=dtype
    ).requires_grad_()
    bias = torch.empty(1, device=gpu_device, dtype=torch.float32).uniform_(-10, 2)
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    actual = module(query, key, value, bias)
    expected = sigmoid_reference(query_ref, key_ref, value_ref, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
    upstream = 0.1 * torch.randn_like(actual)
    _backward_compare(
        actual,
        expected,
        upstream,
        [(query, query_ref), (key, key_ref), (value, value_ref)],
        rtol=1e-1,
        atol=1e-1,
    )


def _random_block_mask(batch, heads_kv, seqlen, block_size, device, ratio=0.5):
    blocks = (seqlen + block_size - 1) // block_size
    valid = math.ceil(seqlen * (1 - ratio) / block_size)
    mask = torch.zeros(batch, heads_kv, blocks, dtype=torch.bool, device=device)
    for batch_idx in range(batch):
        for head_idx in range(heads_kv):
            mask[batch_idx, head_idx, torch.randperm(blocks, device=device)[:valid]] = (
                True
            )
    return mask


def test_legacy_sparse_gqa_decode(gpu_device, seed):
    batch, heads, groups, query_len, kv_len, dim, dim_value = (
        8,
        32,
        8,
        1,
        2048,
        128,
        128,
    )
    block_size = 32
    module = sparse_gqa_decode(
        batch, heads, groups, kv_len, dim, dim_value, BLOCK=block_size
    )
    query = torch.randn(
        batch, query_len, heads, dim, device=gpu_device, dtype=torch.float16
    )
    key = torch.randn(
        batch, kv_len, groups, dim, device=gpu_device, dtype=torch.float16
    )
    value = torch.randn(
        batch, kv_len, groups, dim_value, device=gpu_device, dtype=torch.float16
    )
    cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int32, device=gpu_device)
    block_mask = _random_block_mask(batch, groups, kv_len, block_size, gpu_device)
    actual = module(
        query, key, value, block_mask=block_mask, cache_seqlens=cache_seqlens
    )
    expected = sparse_reference(
        query, key, value, block_mask, cache_seqlens, block_size
    )
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)


def test_legacy_retnet_recurrent(gpu_device, seed):
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 128
    dtype = torch.bfloat16
    module = retnet_recurrent(batch, heads, seqlen, dim, dim_value, dtype=dtype)
    module = _compile_stateful(
        module,
        query_heads=heads,
        key_heads=heads,
        value_heads=heads,
        state_heads=heads,
        key_dim=dim,
        value_dim=dim_value,
        backward=True,
    )
    query = torch.randn(
        batch, heads, seqlen, dim, device=gpu_device, dtype=dtype
    ).requires_grad_()
    key = (0.1 * torch.randn_like(query)).requires_grad_()
    value = torch.randn(
        batch, heads, seqlen, dim_value, device=gpu_device, dtype=dtype
    ).requires_grad_()
    gate = 1 - torch.exp2(
        -5 - torch.arange(heads, device=gpu_device, dtype=torch.float32)
    )
    gate = gate[None, :, None].expand(batch, heads, seqlen).contiguous()
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    actual = module(query=query, key=key, value=value, gate=gate).outputs["output"]
    expected = retnet_reference(query_ref, key_ref, value_ref)
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
    upstream = 0.1 * torch.randn_like(actual)
    _backward_compare(
        actual,
        expected,
        upstream,
        [(query, query_ref), (key, key_ref), (value, value_ref)],
        rtol=1e-1,
        atol=1e-1,
    )


@pytest.mark.parametrize("batch,seqlen", [(1, 2048), (32, 512), (32, 1024)])
def test_legacy_relu(batch, seqlen, gpu_device, seed):
    heads, dim, dim_value = 6, 64, 64
    dtype = torch.float16
    module = relu_attention(
        batch, heads, seqlen, dim, dim_value, dtype=dtype, tune=True
    )
    query = torch.randn(
        batch, seqlen, heads, dim, device=gpu_device, dtype=dtype
    ).requires_grad_()
    key = (0.5 * torch.randn_like(query)).requires_grad_()
    value = (
        0.5
        * torch.randn(batch, seqlen, heads, dim_value, device=gpu_device, dtype=dtype)
    ).requires_grad_()
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    actual = module(query, key, value)
    expected = relu_reference(query_ref, key_ref, value_ref)
    for actual_tensor, expected_tensor in ((actual, expected),):
        assert_close(
            actual_tensor, expected_tensor, rtol=1e-1, atol=1e-1, mismatch_ratio=1e-3
        )
    upstream = 0.1 * torch.randn_like(actual)
    _backward_compare(
        actual,
        expected,
        upstream,
        [(query, query_ref), (key, key_ref), (value, value_ref)],
        rtol=1e-1,
        atol=1e-1,
        mismatch_ratio=1e-3,
    )


def test_legacy_mla_decode(gpu_device, seed):
    batch, heads, kv_len, dim, dim_value, kv_heads = 8, 128, 2048, 576, 512, 1
    dtype = torch.float16
    module = mla_decode(
        batch,
        heads,
        kv_len,
        dim,
        dim_value,
        HK=kv_heads,
        HV=kv_heads,
        tune=True,
        dtype=dtype,
    )
    query = torch.randn(batch, 1, heads, dim_value, device=gpu_device, dtype=dtype)
    query_pe = torch.randn(
        batch, 1, heads, dim - dim_value, device=gpu_device, dtype=dtype
    )
    key_value = torch.randn(
        batch, kv_len, kv_heads, dim_value, device=gpu_device, dtype=dtype
    )
    key_pe = torch.randn(
        batch, kv_len, kv_heads, dim - dim_value, device=gpu_device, dtype=dtype
    )
    actual = module(query, query_pe, key_value, key_pe)
    expected = mla_decode_reference(query, query_pe, key_value, key_pe)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
