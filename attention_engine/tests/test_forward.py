import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

from benchmark.bench_utils import assert_close
from examples.mha import causal_softmax_attention
from examples.mha_decode import softmax_attention_decode
from examples.reluattn import relu_attention
from examples.sigmoid_attn import sigmoid_attention
from examples.sparse_gqa_decode import sparse_gqa_decode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="forward kernel tests require CUDA/HIP GPU",
)

DEVICE = "cuda"
DTYPE = torch.float16
RTOL_STRICT = 1e-2
ATOL_STRICT = 1e-2
RTOL_LOOSE = 1e-1
ATOL_LOOSE = 1e-1


def _seed() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def _causal_softmax_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    dim = query.shape[-1]
    num_head_groups = query.shape[2] // key.shape[2]
    if softmax_scale is None:
        softmax_scale = 1 / dim**0.5

    query = rearrange(query, "b s (h g) d -> b s g h d", g=num_head_groups)
    scores = einsum(query, key, "b s g h d, b t h d -> b g h s t")
    seqlenq = query.shape[1]
    seqlenk = key.shape[1]
    mask = torch.tril(torch.ones(seqlenq, seqlenk, device=scores.device))
    mask = mask.unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attention = F.softmax(scores * softmax_scale, dim=-1)

    out = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    return rearrange(out, "b g h s d -> b s (h g) d")


def _decode_softmax_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    dim = query.shape[-1]
    num_head_groups = query.shape[2] // key.shape[2]
    if softmax_scale is None:
        softmax_scale = 1 / dim**0.5

    query = rearrange(query, "b s (h g) d -> b s g h d", g=num_head_groups)
    scores = einsum(query, key, "b s g h d, b t h d -> b g h s t")
    attention = F.softmax(scores * softmax_scale, dim=-1)

    out = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    return rearrange(out, "b g h s d -> b s (h g) d")


def test_causal_softmax_attention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = causal_softmax_attention(
        batch,
        heads,
        seqlen,
        dim,
        dim_value,
        dtype=DTYPE,
        tune=False,
    )

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE)

    expected = _causal_softmax_ref(query, key, value)
    actual = module(query, key, value)

    torch.testing.assert_close(actual, expected, rtol=RTOL_STRICT, atol=ATOL_STRICT)


def test_softmax_decode_forward_matches_reference():
    _seed()
    batch, heads, seqlen, kv_len, dim, dim_value = 1, 2, 1, 128, 64, 64
    module = softmax_attention_decode(
        batch, heads, seqlen, kv_len, dim, dim_value, dtype=DTYPE
    )

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, kv_len, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, kv_len, heads, dim_value, device=DEVICE, dtype=DTYPE)

    expected = _decode_softmax_ref(query, key, value)
    actual = module(query, key, value)

    torch.testing.assert_close(actual, expected, rtol=RTOL_STRICT, atol=ATOL_STRICT)


def test_sigmoid_attention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = sigmoid_attention(batch, heads, seqlen, dim, dim_value, tune=False)

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE)
    sigmoid_bias = torch.tensor([0.25], device=DEVICE, dtype=torch.float32)

    num_head_groups = query.shape[2] // key.shape[2]
    grouped_query = rearrange(query, "b s (h g) d -> b s g h d", g=num_head_groups)
    scores = einsum(grouped_query, key, "b s g h d, b t h d -> b g h s t")
    mask = torch.tril(
        torch.ones(grouped_query.shape[1], key.shape[1], device=scores.device)
    )
    mask = mask.unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    scores = scores + sigmoid_bias
    attention = (torch.tanh(scores * 0.5) + 1) * 0.5
    expected = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    expected = rearrange(expected, "b g h s d -> b s (h g) d")

    actual = module(query, key, value, sigmoid_bias)

    torch.testing.assert_close(actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_relu_attention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = relu_attention(
        batch, heads, seqlen, dim, dim_value, dtype=DTYPE, tune=False
    )

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = 0.5 * torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = 0.5 * torch.randn(
        batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE
    )

    qk = torch.einsum("bqhd,bkhd->bhqk", query, key)
    qk = qk / (dim**0.5)
    qk = F.relu(qk)
    expected = torch.einsum("bhqk,bkhd->bqhd", qk, value)

    actual = module(query, key, value)

    assert_close(
        actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE, mismatch_ratio=1e-3
    )


def _sparse_gqa_decode_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    query = query.squeeze(1)
    batch, heads, dim = query.shape
    heads_kv = key.shape[2]
    num_blocks = block_mask.shape[-1]

    num_head_groups = heads // heads_kv
    scale = dim**0.5
    key = rearrange(key, "b n h d -> b h n d")
    value = rearrange(value, "b n h d -> b h n d")
    query = rearrange(query, "b (h g) d -> b g h d", g=num_head_groups)

    scores = einsum(query, key, "b g h d, b h s d -> b g h s")

    sparse_mask = torch.zeros_like(scores)
    for batch_idx in range(batch):
        for head_idx in range(heads_kv):
            for block_idx in range(num_blocks):
                if block_mask[batch_idx, head_idx, block_idx]:
                    start = block_idx * block_size
                    end = (block_idx + 1) * block_size
                    sparse_mask[batch_idx, :, head_idx, start:end] = 1

    scores = scores.masked_fill(sparse_mask == 0, float("-inf"))

    range_len = torch.arange(scores.shape[-1], device=query.device).unsqueeze(0)
    cache_seqlens_expanded = cache_seqlens.unsqueeze(1)
    pad_mask = range_len >= cache_seqlens_expanded
    pad_mask = pad_mask[:, None, None, :]
    scores = scores.masked_fill(pad_mask, float("-inf"))
    attention = F.softmax(scores / scale, dim=-1)

    out = einsum(attention, value, "b g h s, b h s d -> b g h d")
    return rearrange(out, "b g h d -> b (h g) d")


def _deterministic_block_mask(
    batch: int,
    heads_kv: int,
    max_cache_seqlen: int,
    block_size: int,
    device: str,
) -> torch.Tensor:
    num_blocks = (max_cache_seqlen + block_size - 1) // block_size
    block_mask = torch.zeros(
        (batch, heads_kv, num_blocks), dtype=torch.bool, device=device
    )
    block_mask[:, :, 0] = True
    if num_blocks > 1:
        block_mask[:, :, -1] = True
    return block_mask


def test_sparse_gqa_decode_forward_matches_reference():
    _seed()
    batch, heads, groups, seqlen, dim, dim_value, block = 1, 4, 2, 128, 64, 64, 32
    module = sparse_gqa_decode(
        batch, heads, groups, seqlen, dim, dim_value, dtype=DTYPE, BLOCK=block
    )

    query = torch.randn(batch, 1, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, groups, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, groups, dim_value, device=DEVICE, dtype=DTYPE)
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=DEVICE)
    block_mask = _deterministic_block_mask(batch, groups, seqlen, block, DEVICE)

    expected = _sparse_gqa_decode_ref(
        query, key, value, block_mask, cache_seqlens, block
    )
    actual = module(
        query, key, value, block_mask=block_mask, cache_seqlens=cache_seqlens
    )

    torch.testing.assert_close(actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)
