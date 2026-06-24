import math

import pytest
import torch
import torch.nn.functional as F
from benchmark.bench_utils import assert_close
from einops import einsum, rearrange, repeat

from examples.gated_retention import gated_retention
from examples.mamba2 import mamba2
from examples.mha import causal_softmax_attention
from examples.mha_decode import softmax_attention_decode
from examples.mha_v2 import causal_softmax_attention as causal_softmax_attention_v2
from examples.mla_decode import mla_decode
from examples.mla_decode_v2 import mla_decode as mla_decode_v2
from examples.reluattn import relu_attention
from examples.reluattn_v2 import relu_attention as relu_attention_v2
from examples.retention_parallel import retention_parallel
from examples.retnet_recurrent import retnet_recurrent
from examples.sigmoid_attn import sigmoid_attention
from examples.sigmoid_attn_v2 import sigmoid_attention as sigmoid_attention_v2
from examples.sparse_gqa_decode import sparse_gqa_decode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="forward kernel tests require CUDA/HIP GPU",
)

DEVICE = "cuda"
DTYPE = torch.float16
LINEAR_DTYPE = torch.bfloat16
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


def _sigmoid_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sigmoid_bias: torch.Tensor,
) -> torch.Tensor:
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
    return rearrange(expected, "b g h s d -> b s (h g) d")


def _relu_ref(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    dim = query.shape[-1]
    qk = torch.einsum("bqhd,bkhd->bhqk", query, key)
    qk = qk / (dim**0.5)
    qk = F.relu(qk)
    return torch.einsum("bhqk,bkhd->bqhd", qk, value)


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


def _mla_decode_ref(
    query: torch.Tensor,
    query_pe: torch.Tensor,
    key_value: torch.Tensor,
    key_pe: torch.Tensor,
) -> torch.Tensor:
    query = query.squeeze(1)
    query_pe = query_pe.squeeze(1)
    dim = query.shape[-1]
    pe_dim = query_pe.shape[-1]
    num_head_groups = query.shape[1] // key_value.shape[2]
    scale = (dim + pe_dim) ** 0.5

    query = rearrange(query, "b (h g) d -> b g h d", g=num_head_groups)
    query_pe = rearrange(query_pe, "b (h g) d -> b g h d", g=num_head_groups)
    key_value = rearrange(key_value, "b n h d -> b h n d")
    key_pe = rearrange(key_pe, "b n h d -> b h n d")

    full_query = torch.concat([query, query_pe], dim=-1)
    full_key = torch.concat([key_value, key_pe], dim=-1)
    scores = einsum(full_query, full_key, "b g h d, b h s d -> b g h s")
    attention = F.softmax(scores / scale, dim=-1)
    out = einsum(attention, key_value, "b g h s, b h s d -> b g h d")
    out = rearrange(out, "b g h d -> b (h g) d")
    return out.unsqueeze(1)


def _retention_parallel_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    qk = torch.einsum("bqhd,bkhd->bhqk", query, key)
    masked_scores = qk * mask
    rowsum = masked_scores.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.einsum("bhqk,bkhd->bqhd", masked_scores / rowsum, value)


def _naive_chunk_simple_gla(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    chunk_size: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    gate = gate.to(torch.float32)
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5

    total_steps = query.shape[-2]
    pad_len = (chunk_size - (total_steps % chunk_size)) % chunk_size
    if pad_len > 0:
        query = F.pad(query, (0, 0, 0, pad_len))
        key = F.pad(key, (0, 0, 0, pad_len))
        value = F.pad(value, (0, 0, 0, pad_len))
        gate = F.pad(gate, (0, pad_len))

    batch, heads, padded_steps, dim = query.shape
    dim_value = value.shape[-1]
    query = query * scale
    query, key, value, decay = map(
        lambda tensor: rearrange(tensor, "b h (n c) d -> b h n c d", c=chunk_size),
        [query, key, value, gate.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    local_mask = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()
    ).tril()
    state = key.new_zeros(batch, heads, dim, dim_value)
    output = torch.zeros_like(value)
    for chunk_idx in range(padded_steps // chunk_size):
        query_chunk = query[:, :, chunk_idx]
        key_chunk = key[:, :, chunk_idx]
        value_chunk = value[:, :, chunk_idx]
        attn = query_chunk @ key_chunk.transpose(-1, -2)
        attn = attn * local_mask[:, :, chunk_idx]
        carry = (query_chunk * decay[:, :, chunk_idx, :, None].exp()) @ state
        output[:, :, chunk_idx] = carry + attn @ value_chunk
        end_decay = decay[:, :, chunk_idx, -1, None]
        key_scale = (end_decay - decay[:, :, chunk_idx]).exp()[..., None]
        state = (
            state * end_decay.exp()[..., None]
            + (key_chunk * key_scale).transpose(-1, -2) @ value_chunk
        )

    return rearrange(output, "b h n c d -> b h (n c) d")[:, :, :total_steps]


def _retnet_recurrent_ref(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    orig_type = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    _, heads, seq_len, dim = query.shape
    slope = (
        1
        - query.new_tensor(2.0, dtype=torch.float).pow(
            -5.0 - query.new_tensor(range(heads), dtype=torch.float)
        )
    ).log2()
    positions = query.new_tensor(range(seq_len), dtype=torch.float)
    decay = torch.exp2((positions.unsqueeze(-1) - positions) * slope.view(-1, 1, 1))
    decay = decay * positions.unsqueeze(-1).ge(positions)
    scores = torch.einsum(
        "bhqd,bhkd,hqk->bhqk", query * dim**-0.5, key, decay.to(query.dtype)
    )
    return torch.einsum("bhqk,bhkd->bhqd", scores, value).to(orig_type)


def _mamba2_ref(
    value: torch.Tensor,
    delta_t: torch.Tensor,
    a_param: torch.Tensor,
    key: torch.Tensor,
    query: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    def chunk_state_ref(
        b_tensor: torch.Tensor,
        x_tensor: torch.Tensor,
        dt_tensor: torch.Tensor,
        d_a_cumsum: torch.Tensor,
    ) -> torch.Tensor:
        batch, seqlen, nheads, headdim = x_tensor.shape
        dstate = b_tensor.shape[-1]
        _, _, nchunks, local_chunk = dt_tensor.shape
        ngroups = b_tensor.shape[2]
        assert seqlen == nchunks * local_chunk
        assert nheads % ngroups == 0
        expanded_b = repeat(b_tensor, "b l g d -> b l (g h) d", h=nheads // ngroups)
        x_tensor = rearrange(x_tensor, "b (c l) h p -> b c l h p", l=local_chunk)
        expanded_b = rearrange(expanded_b, "b (c l) h n -> b c l h n", l=local_chunk)
        decay_states = torch.exp(d_a_cumsum[:, :, :, -1:] - d_a_cumsum)
        return torch.einsum(
            "bclhn,bhcl,bhcl,bclhp->bchpn",
            expanded_b.to(x_tensor.dtype),
            decay_states.to(x_tensor.dtype),
            dt_tensor.to(x_tensor.dtype),
            x_tensor,
        )

    def state_passing_ref(
        states: torch.Tensor, d_a_chunk_cumsum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_states = torch.zeros_like(states[:, 0])
        states = torch.cat(
            [rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1
        )
        d_a_chunk_cumsum = F.pad(d_a_chunk_cumsum, (1, 0))
        d_a_chunk_cumsum = torch.cumsum(d_a_chunk_cumsum, dim=-1)
        nchunks = d_a_chunk_cumsum.shape[-1]
        dt_chunk_segment_sum = (
            d_a_chunk_cumsum[:, :, :, None] - d_a_chunk_cumsum[:, :, None, :]
        )
        decay_chunk = torch.exp(dt_chunk_segment_sum)
        causal_mask = torch.tril(
            torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0
        )
        decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
        out = torch.einsum(
            "bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states
        )
        return out[:, :-1], out[:, -1]

    def chunk_scan_ref(
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        x_tensor: torch.Tensor,
        dt_tensor: torch.Tensor,
        d_a_cumsum: torch.Tensor,
        prev_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, seqlen, nheads, _ = x_tensor.shape
        _, _, ngroups, _ = b_tensor.shape
        _, _, nchunks, local_chunk = dt_tensor.shape
        assert seqlen == nchunks * local_chunk
        expanded_b = repeat(b_tensor, "b l g d -> b l (g h) d", h=nheads // ngroups)
        expanded_c = repeat(c_tensor, "b l g d -> b l (g h) d", h=nheads // ngroups)
        cb = torch.einsum(
            "bclhn,bcshn->bchls",
            rearrange(expanded_c, "b (c l) h n -> b c l h n", c=nchunks),
            rearrange(expanded_b, "b (c s) h n -> b c s h n", c=nchunks),
        )
        dt_segment_sum = d_a_cumsum[:, :, :, :, None] - d_a_cumsum[:, :, :, None, :]
        decay = torch.exp(dt_segment_sum)
        scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
        causal_mask = torch.tril(
            torch.ones(local_chunk, local_chunk, device=x_tensor.device, dtype=bool),
            diagonal=0,
        )
        scores_decay = scores_decay.masked_fill(~causal_mask, 0)
        out = torch.einsum(
            "bchls,bhcs,bcshp->bclhp",
            scores_decay.to(x_tensor.dtype),
            dt_tensor.to(x_tensor.dtype),
            rearrange(x_tensor, "b (c s) h p -> b c s h p", c=nchunks),
        )
        state_decay_out = torch.exp(rearrange(d_a_cumsum, "b h c l -> b c l h 1"))
        out_prev = torch.einsum(
            "bclhn,bchpn->bclhp",
            rearrange(expanded_c, "b (c l) h n -> b c l h n", c=nchunks),
            prev_states.to(expanded_c.dtype),
        )
        out = out + out_prev * state_decay_out
        return rearrange(out, "b c l h p -> b (c l) h p")

    dt_chunks = rearrange(delta_t, "b (c l) h -> b h c l", l=chunk_size).float()
    d_a = dt_chunks * rearrange(a_param, "h -> h 1 1")
    d_a_cumsum = torch.cumsum(d_a, dim=-1)
    states = chunk_state_ref(key, value, dt_chunks, d_a_cumsum)
    state_dtype = states.dtype
    if state_dtype not in (torch.float32, torch.float64):
        states = states.to(torch.float32)
    states = rearrange(
        state_passing_ref(
            rearrange(states, "... p n -> ... (p n)"), d_a_cumsum[:, :, :, -1]
        )[0],
        "... (p n) -> ... p n",
        n=key.shape[-1],
    )
    states = states.to(state_dtype)
    return chunk_scan_ref(key, query, value, dt_chunks, d_a_cumsum, states)


def test_causal_softmax_attention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = causal_softmax_attention(
        batch, heads, seqlen, dim, dim_value, dtype=DTYPE, tune=False
    )

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE)

    expected = _causal_softmax_ref(query, key, value)
    actual = module(query, key, value)

    torch.testing.assert_close(actual, expected, rtol=RTOL_STRICT, atol=ATOL_STRICT)


def test_causal_softmax_attention_v2_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = causal_softmax_attention_v2(
        batch, heads, seqlen, dim, dim_value, dtype=DTYPE
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

    expected = _sigmoid_ref(query, key, value, sigmoid_bias)
    actual = module(query, key, value, sigmoid_bias)

    torch.testing.assert_close(actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_sigmoid_attention_v2_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = sigmoid_attention_v2(batch, heads, seqlen, dim, dim_value, dtype=DTYPE)

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE)
    sigmoid_bias = torch.tensor([0.25], device=DEVICE, dtype=torch.float32)

    expected = _sigmoid_ref(query, key, value, sigmoid_bias)
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

    expected = _relu_ref(query, key, value)
    actual = module(query, key, value)

    assert_close(
        actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE, mismatch_ratio=1e-3
    )


def test_relu_attention_v2_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = relu_attention_v2(batch, heads, seqlen, dim, dim_value, dtype=DTYPE)

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = 0.5 * torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = 0.5 * torch.randn(
        batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE
    )

    expected = _relu_ref(query, key, value)
    actual = module(query, key, value)

    assert_close(
        actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE, mismatch_ratio=1e-3
    )


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


def test_mla_decode_forward_matches_reference():
    _seed()
    batch, heads, kv_heads, seqlen, dim, dim_value = 1, 4, 1, 128, 80, 64
    module = mla_decode(
        batch,
        heads,
        seqlen,
        dim,
        dim_value,
        HK=kv_heads,
        HV=kv_heads,
        dtype=DTYPE,
        tune=False,
    )

    query = torch.randn(batch, 1, heads, dim_value, device=DEVICE, dtype=DTYPE)
    query_pe = torch.randn(batch, 1, heads, dim - dim_value, device=DEVICE, dtype=DTYPE)
    key_value = torch.randn(
        batch, seqlen, kv_heads, dim_value, device=DEVICE, dtype=DTYPE
    )
    key_pe = torch.randn(
        batch, seqlen, kv_heads, dim - dim_value, device=DEVICE, dtype=DTYPE
    )

    expected = _mla_decode_ref(query, query_pe, key_value, key_pe)
    actual = module(query, query_pe, key_value, key_pe)

    torch.testing.assert_close(actual, expected, rtol=RTOL_STRICT, atol=ATOL_STRICT)


def test_mla_decode_v2_forward_matches_reference():
    _seed()
    batch, heads, kv_heads, seqlen, dim, dim_value = 1, 4, 1, 128, 80, 64
    module = mla_decode_v2(
        batch,
        heads,
        seqlen,
        dim,
        dim_value,
        HK=kv_heads,
        HV=kv_heads,
        SQ=1,
        dtype=LINEAR_DTYPE,
    )

    query = torch.randn(batch, 1, heads, dim_value, device=DEVICE, dtype=LINEAR_DTYPE)
    query_pe = torch.randn(
        batch, 1, heads, dim - dim_value, device=DEVICE, dtype=LINEAR_DTYPE
    )
    key_value = torch.randn(
        batch, seqlen, kv_heads, dim_value, device=DEVICE, dtype=LINEAR_DTYPE
    )
    key_pe = torch.randn(
        batch, seqlen, kv_heads, dim - dim_value, device=DEVICE, dtype=LINEAR_DTYPE
    )

    expected = _mla_decode_ref(query, query_pe, key_value, key_pe)
    actual = module(query, query_pe, key_value, key_pe)

    torch.testing.assert_close(actual, expected, rtol=RTOL_STRICT, atol=ATOL_STRICT)


def test_retention_parallel_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 64, 32, 32
    module = retention_parallel(
        batch, heads, seqlen, dim, dim_value, dtype=DTYPE, tune=False
    )

    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE)
    mask = (
        torch.rand(1, heads, seqlen, seqlen, device=DEVICE, dtype=DTYPE)
        .tril()
        .contiguous()
    )

    expected = _retention_parallel_ref(query, key, value, mask)
    actual = module(query, key, value, mask)

    torch.testing.assert_close(actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_gated_retention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 64, 32, 32
    module = gated_retention(
        batch, heads, seqlen, dim, dim_value, dtype=LINEAR_DTYPE, tune=False
    )

    query = torch.randn(batch, heads, seqlen, dim, device=DEVICE, dtype=LINEAR_DTYPE)
    key = torch.randn(batch, heads, seqlen, dim, device=DEVICE, dtype=LINEAR_DTYPE)
    gate = F.logsigmoid(
        torch.randn(batch, heads, seqlen, device=DEVICE, dtype=torch.float32)
    ).clamp_min(-5)
    value = torch.randn(
        batch, heads, seqlen, dim_value, device=DEVICE, dtype=LINEAR_DTYPE
    )

    expected = _naive_chunk_simple_gla(query, key, value, gate).to(LINEAR_DTYPE)
    actual = module(query, key, value, gate)

    torch.testing.assert_close(actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_retnet_recurrent_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 64, 32, 32
    module = retnet_recurrent(
        batch, heads, seqlen, dim, dim_value, dtype=LINEAR_DTYPE, tune=False
    )

    query = torch.randn(batch, heads, seqlen, dim, device=DEVICE, dtype=LINEAR_DTYPE)
    key = 0.1 * torch.randn(
        batch, heads, seqlen, dim, device=DEVICE, dtype=LINEAR_DTYPE
    )
    gate = 1 - torch.exp2(-5 - torch.arange(heads, device=DEVICE, dtype=torch.float32))
    gate = gate[None, :, None].expand(batch, heads, seqlen).contiguous()
    value = torch.randn(
        batch, heads, seqlen, dim_value, device=DEVICE, dtype=LINEAR_DTYPE
    )

    expected = _retnet_recurrent_ref(query, key, value)
    actual = module(query, key, value, gate)

    torch.testing.assert_close(actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_mamba2_forward_matches_reference():
    _seed()
    batch, query_heads, key_heads, value_heads, seqlen, dim, dim_value = (
        1,
        1,
        1,
        2,
        64,
        32,
        16,
    )
    module = mamba2(
        batch,
        query_heads,
        seqlen,
        dim,
        dim_value,
        HK=key_heads,
        HV=value_heads,
        dtype=LINEAR_DTYPE,
        tune=False,
    )

    query = torch.randn(
        batch, seqlen, query_heads, dim, device=DEVICE, dtype=LINEAR_DTYPE
    )
    key = 0.5 * torch.randn(
        batch, seqlen, key_heads, dim, device=DEVICE, dtype=LINEAR_DTYPE
    )
    value = torch.randn(
        batch, seqlen, value_heads, dim_value, device=DEVICE, dtype=LINEAR_DTYPE
    )
    a_param = 1.5 * torch.rand(value_heads, dtype=LINEAR_DTYPE, device=DEVICE) - 4.0
    dt_seed = 0.7 * torch.rand(
        batch, seqlen, value_heads, dtype=torch.float32, device=DEVICE
    )
    dt_min = 0.001
    dt_max = 0.1
    dt_base = torch.exp(
        torch.rand(value_heads, device=DEVICE, dtype=LINEAR_DTYPE)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt_base = torch.clamp(dt_base, min=1e-4)
    dt_bias = dt_base + torch.log(-torch.expm1(-dt_base))
    delta_t = F.softplus(dt_seed + dt_bias)

    query_ours = query.transpose(1, 2).contiguous()
    key_ours = key.transpose(1, 2).contiguous()
    value_ours = value.transpose(1, 2).contiguous()
    a_ours = a_param[None, :].contiguous()
    delta_t_ours = delta_t.transpose(1, 2).contiguous()

    expected = _mamba2_ref(value, delta_t, a_param, key, query).to(LINEAR_DTYPE)
    actual = module(
        query_ours,
        key_ours,
        value_ours,
        delta_t_ours,
        a_ours,
        delta_t_ours.to(LINEAR_DTYPE),
    )

    assert_close(
        actual.transpose(1, 2),
        expected,
        rtol=RTOL_LOOSE,
        atol=ATOL_LOOSE,
        mismatch_ratio=1e-2,
    )
