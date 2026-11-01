from __future__ import annotations

import pytest
import torch

from examples.mha_v2 import causal_softmax_attention as mha_v2
from examples.mla_decode import mla_decode as mla_v1
from examples.mla_decode_v2 import mla_decode as mla_v2
from examples.reluattn_v2 import relu_attention as relu_v2
from examples.retention_parallel import retention_parallel
from examples.sigmoid_attn_v2 import sigmoid_attention as sigmoid_v2
from examples.sparse_gqa_decode import sparse_gqa_decode

from .reference import (
    causal_softmax,
    mla_decode,
    relu_attention,
    retention_parallel as retention_reference,
    sigmoid_attention,
    sparse_gqa_decode as sparse_reference,
)


pytestmark = [pytest.mark.functional, pytest.mark.gpu, pytest.mark.slow]


def test_mha_v2_forward_contract(gpu_device, seed):
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    dtype = torch.float16
    module = mha_v2(batch, heads, seqlen, dim, dim_value, dtype=dtype)
    query = torch.randn(batch, seqlen, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, seqlen, heads, dim_value, device=gpu_device, dtype=dtype)
    actual = module(query, key, value)
    expected = causal_softmax(query, key, value)
    assert actual.shape == (batch, seqlen, heads, dim_value)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_relu_v2_forward_contract(gpu_device, seed):
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    dtype = torch.float16
    module = relu_v2(batch, heads, seqlen, dim, dim_value, dtype=dtype)
    query = torch.randn(batch, seqlen, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, seqlen, heads, dim_value, device=gpu_device, dtype=dtype)
    actual = module(query, key, value)
    expected = relu_attention(query, key, value)
    assert actual.shape == (batch, seqlen, heads, dim_value)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)


def test_sigmoid_v2_forward_contract(gpu_device, seed):
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    dtype = torch.float16
    module = sigmoid_v2(batch, heads, seqlen, dim, dim_value, dtype=dtype)
    query = torch.randn(batch, seqlen, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, seqlen, heads, dim_value, device=gpu_device, dtype=dtype)
    bias = torch.tensor([0.25], device=gpu_device, dtype=torch.float32)
    actual = module(query, key, value, bias)
    expected = sigmoid_attention(query, key, value, bias)
    assert actual.shape == (batch, seqlen, heads, dim_value)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)


def _mla_inputs(batch, heads, kv_len, dim, dim_value, kv_heads, dtype, device):
    return (
        torch.randn(batch, 1, heads, dim_value, device=device, dtype=dtype),
        torch.randn(batch, 1, heads, dim - dim_value, device=device, dtype=dtype),
        torch.randn(batch, kv_len, kv_heads, dim_value, device=device, dtype=dtype),
        torch.randn(
            batch, kv_len, kv_heads, dim - dim_value, device=device, dtype=dtype
        ),
    )


@pytest.mark.parametrize(
    ("factory", "dtype"),
    [(mla_v1, torch.float16), (mla_v2, torch.bfloat16)],
    ids=["mla-v1", "mla-v2"],
)
def test_mla_v1_v2_forward_contract(factory, dtype, gpu_device, seed):
    batch, heads, kv_len, dim, dim_value, kv_heads = 1, 128, 128, 576, 512, 1
    module = factory(
        batch,
        heads,
        kv_len,
        dim,
        dim_value,
        HK=kv_heads,
        HV=kv_heads,
        dtype=dtype,
    )
    query, query_pe, key_value, key_pe = _mla_inputs(
        batch, heads, kv_len, dim, dim_value, kv_heads, dtype, gpu_device
    )
    actual = module(query, query_pe, key_value, key_pe)
    expected = mla_decode(query, query_pe, key_value, key_pe)
    assert actual.shape == (batch, 1, heads, dim_value)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_retention_parallel_custom_mask_contract(gpu_device, seed):
    batch, heads, seqlen, dim, dim_value = 1, 2, 64, 32, 48
    dtype = torch.float16
    module = retention_parallel(batch, heads, seqlen, dim, dim_value, dtype=dtype)
    query = torch.randn(batch, seqlen, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, seqlen, heads, dim_value, device=gpu_device, dtype=dtype)
    mask = torch.tril(
        torch.ones(1, heads, seqlen, seqlen, device=gpu_device, dtype=dtype)
    )
    actual = module(query, key, value, mask)
    expected = retention_reference(query, key, value, mask)
    assert actual.shape == (batch, seqlen, heads, dim_value)
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)


def test_sparse_gqa_deterministic_mask_contract(gpu_device, seed):
    batch, heads, groups, kv_len, dim, dim_value, block_size = 1, 4, 2, 64, 32, 48, 32
    dtype = torch.float16
    module = sparse_gqa_decode(
        batch, heads, groups, kv_len, dim, dim_value, BLOCK=block_size
    )
    query = torch.randn(batch, 1, heads, dim, device=gpu_device, dtype=dtype)
    key = torch.randn(batch, kv_len, groups, dim, device=gpu_device, dtype=dtype)
    value = torch.randn(
        batch, kv_len, groups, dim_value, device=gpu_device, dtype=dtype
    )
    block_mask = torch.zeros(batch, groups, 2, dtype=torch.bool, device=gpu_device)
    block_mask[:, :, 0] = True
    block_mask[:, :, 1] = True
    cache_seqlens = torch.full((batch,), kv_len, dtype=torch.int32, device=gpu_device)
    actual = module(
        query, key, value, block_mask=block_mask, cache_seqlens=cache_seqlens
    )
    expected = sparse_reference(
        query, key, value, block_mask, cache_seqlens, block_size
    )
    assert actual.shape == (batch, heads, dim_value)
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
