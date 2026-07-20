import pytest
import torch
import torch.nn.functional as F
from attn_engine import AttentionEngine, OnlineFunc
from benchmark.bench_utils import assert_close
from core import CustomIO, SymbolScalar, Var, meta_tensor
from einops import einsum, rearrange

pytestmark = [
    pytest.mark.functional,
    pytest.mark.gpu,
    pytest.mark.usefixtures("gpu_device", "seed"),
]

DEVICE = "cuda"
DTYPE = torch.float16
RTOL_STRICT = 1e-2
ATOL_STRICT = 1e-2
RTOL_LOOSE = 1e-1
ATOL_LOOSE = 1e-1


def _seed() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


class _OnlineSoftmax(OnlineFunc):
    def __init__(self):
        online_rowscales = {
            "m": SymbolScalar("m", Var("-inf")),
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "lse": SymbolScalar("lse", Var("0.0")),
        }
        super().__init__(online_rowscales, final_rowscales, CustomIO())

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):
        m = online_rowscales["m"]
        r = online_rowscales["r"]
        m_new = m.max(scores.get_reduce("max"))
        scale_tmp = (m - m_new).exp()
        r = r * scale_tmp
        scores = (scores - m_new).exp()
        r = r + scores.get_reduce("sum")
        new_online_rowscales = {
            "m": m_new,
            "r": r,
        }
        return scores, new_online_rowscales, scale_tmp

    @staticmethod
    def combine(final_rowscales):
        lse = final_rowscales["lse"]
        lse_max = lse.get_reduce("max")
        row_sum = (lse - lse_max).exp()
        row_sum_sum = row_sum.get_reduce("sum")
        lse_sum = row_sum_sum.log() + lse_max
        return (lse - lse_sum).exp()

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        o_new = o / online_rowscales["r"]
        lse = online_rowscales["r"].log() + online_rowscales["m"]
        return o_new, {"lse": lse}

    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        lse = final_rowscales["lse"]
        return (scores - lse).exp()

    @staticmethod
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        return (dp - doosum_rowscales) * scores


class _OnlineIdentity(OnlineFunc):
    def __init__(self):
        super().__init__({}, {}, CustomIO())

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):
        return scores, online_rowscales, SymbolScalar("o_scale", Var("1"))

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        return o, {}

    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        return scores

    @staticmethod
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        return dp


class _OnlineRetention(OnlineFunc):
    def __init__(self):
        online_rowscales = {
            "r_wo_clamp": SymbolScalar("r_wo_clamp", Var("0.0")),
            "r": SymbolScalar("r", Var("0.0")),
        }
        final_rowscales = {
            "r": SymbolScalar("r", Var("0.0")),
        }
        super().__init__(online_rowscales, final_rowscales, CustomIO())

    @staticmethod
    def online_fwd(scores, online_rowscales, b, h, q_idx):
        r_wo_clamp = online_rowscales["r_wo_clamp"]
        r = online_rowscales["r"]
        r_wo_clamp = r_wo_clamp + scores.get_reduce("abssum")
        r_new = r_wo_clamp.max(1.0)
        scores = scores / r_new
        new_online_rowscales = {
            "r_wo_clamp": r_wo_clamp,
            "r": r_new,
        }
        return scores, new_online_rowscales, r / r_new

    @staticmethod
    def online_fwd_epilogue(o, online_rowscales, b, h, q_idx):
        return o, {"r": online_rowscales["r"]}

    @staticmethod
    def forward(scores, final_rowscales, b, h, q_idx, kv_idx):
        return scores / final_rowscales["r"]

    @staticmethod
    def backward(dp, scores, final_rowscales, doosum_rowscales, b, h, q_idx, kv_idx):
        return dp / final_rowscales["r"]


def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _build_causal_softmax_attention(B, H, S, D, DV, dtype=DTYPE):
    softmax_scale = 1 / D**0.5

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * softmax_scale

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({}),
        score_mod=score_mod,
        mask_mod=_causal_mask,
        online_func=_OnlineSoftmax(),
        infer_mask=True,
    )


def _build_causal_softmax_gqa_attention(B, H, G, S, D, DV, dtype=DTYPE):
    softmax_scale = 1 / D**0.5

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * softmax_scale

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, G, S, D, dtype=dtype),
        meta_tensor(B, G, S, DV, dtype=dtype),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({}),
        score_mod=score_mod,
        mask_mod=_causal_mask,
        online_func=_OnlineSoftmax(),
        infer_mask=True,
    )


def _build_softmax_decode_attention(B, H, S, KV, D, DV, dtype=DTYPE):
    softmax_scale = 1 / D**0.5

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * softmax_scale

    qkv_meta = (
        meta_tensor(B, H, ((S + 127) // 128) * 128, D, dtype=dtype),
        meta_tensor(B, H, KV, D, dtype=dtype),
        meta_tensor(B, H, KV, DV, dtype=dtype),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({}),
        score_mod=score_mod,
        mask_mod=None,
        online_func=_OnlineSoftmax(),
    )


def _build_sigmoid_attention(B, H, S, D, DV):
    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        softmax_bias = custom_fwd_inputs.input_tensors["softmax_bias"]
        score = score + softmax_bias
        return ((score * 0.5).tanh() + 1) * 0.5

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, D, dtype=torch.float16),
        meta_tensor(B, H, S, DV, dtype=torch.float16),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({"softmax_bias": (1,)}),
        score_mod=score_mod,
        mask_mod=_causal_mask,
        online_func=_OnlineIdentity(),
    )


def _build_relu_attention(B, H, S, D, DV, dtype=DTYPE):
    scores_scale = 1 / D**0.5

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return (score * scores_scale).max(0)

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({}),
        score_mod=score_mod,
        mask_mod=None,
        online_func=_OnlineIdentity(),
    )


def _build_sparse_gqa_decode_attention(B, H, G, S, D, DV, BLOCK=32, dtype=DTYPE):
    softmax_scale = 1 / D**0.5

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * softmax_scale

    qkv_meta = (
        meta_tensor(B, H, 1, D, dtype=dtype),
        meta_tensor(B, G, S, D, dtype=dtype),
        meta_tensor(B, G, S, DV, dtype=dtype),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({}),
        score_mod=score_mod,
        mask_mod=None,
        online_func=_OnlineSoftmax(),
        extern_block_mask=True,
        use_varlen=True,
        infer_mask_block_N=BLOCK,
    )


def _build_retention_parallel_attention(B, H, S, D, DV, dtype=DTYPE):
    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        mask = custom_fwd_inputs.input_tensors["mask"]
        return score * mask

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({"mask": (1, "heads", "seq_len", "seq_len_kv")}),
        score_mod=score_mod,
        mask_mod=_causal_mask,
        online_func=_OnlineRetention(),
        mask_value="0",
    )


def _build_mla_decode_attention(B, H, SKV, D, DV, HK, HV, dtype=DTYPE):
    softmax_scale = 1 / D**0.5

    def score_mod(score, custom_fwd_inputs, b, h, q_idx, kv_idx):
        return score * softmax_scale

    qkv_meta = (
        meta_tensor(B, H, 1, D, dtype=dtype),
        meta_tensor(B, HK, SKV, D, dtype=dtype),
        meta_tensor(B, HV, SKV, DV, dtype=dtype),
    )
    return AttentionEngine(
        qkv_meta,
        CustomIO({}),
        score_mod=score_mod,
        mask_mod=None,
        online_func=_OnlineSoftmax(),
        kv_shared=True,
    )


def _causal_softmax_ref(query, key, value, softmax_scale=None):
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


def _decode_softmax_ref(query, key, value, softmax_scale=None):
    dim = query.shape[-1]
    num_head_groups = query.shape[2] // key.shape[2]
    if softmax_scale is None:
        softmax_scale = 1 / dim**0.5
    query = rearrange(query, "b s (h g) d -> b s g h d", g=num_head_groups)
    scores = einsum(query, key, "b s g h d, b t h d -> b g h s t")
    attention = F.softmax(scores * softmax_scale, dim=-1)
    out = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    return rearrange(out, "b g h s d -> b s (h g) d")


def _sigmoid_ref(query, key, value, sigmoid_bias):
    num_head_groups = query.shape[2] // key.shape[2]
    grouped_query = rearrange(query, "b s (h g) d -> b s g h d", g=num_head_groups)
    scores = einsum(grouped_query, key, "b s g h d, b t h d -> b g h s t")
    mask = torch.tril(
        torch.ones(grouped_query.shape[1], key.shape[1], device=scores.device)
    )
    mask = mask.unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    scores = scores + sigmoid_bias.to(scores.dtype)
    attention = (torch.tanh(scores * 0.5) + 1) * 0.5
    expected = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    return rearrange(expected, "b g h s d -> b s (h g) d")


def _relu_ref(query, key, value):
    dim = query.shape[-1]
    qk = torch.einsum("bqhd,bkhd->bhqk", query, key)
    qk = qk / (dim**0.5)
    qk = F.relu(qk)
    return torch.einsum("bhqk,bkhd->bqhd", qk, value)


def _sparse_gqa_decode_ref(query, key, value, block_mask, cache_seqlens, block_size):
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


def _deterministic_block_mask(batch, heads_kv, max_cache_seqlen, block_size, device):
    num_blocks = (max_cache_seqlen + block_size - 1) // block_size
    block_mask = torch.zeros(
        (batch, heads_kv, num_blocks), dtype=torch.bool, device=device
    )
    block_mask[:, :, 0] = True
    if num_blocks > 1:
        block_mask[:, :, -1] = True
    return block_mask


def _mla_decode_ref(query, query_pe, key_value, key_pe):
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


def _retention_parallel_ref(query, key, value, mask):
    qk = torch.einsum("bqhd,bkhd->bhqk", query, key)
    qkm = qk * mask
    rowsum = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.einsum("bhqk,bkhd->bqhd", qkm / rowsum, value)


def test_causal_softmax_attention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = _build_causal_softmax_attention(
        batch, heads, seqlen, dim, dim_value, dtype=DTYPE
    )
    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE)
    expected = _causal_softmax_ref(query, key, value)
    actual = module(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=RTOL_STRICT, atol=ATOL_STRICT)


def test_causal_softmax_gqa_forward_matches_reference():
    _seed()
    batch, heads, groups, seqlen, dim, dim_value = 1, 4, 2, 128, 64, 64
    module = _build_causal_softmax_gqa_attention(
        batch, heads, groups, seqlen, dim, dim_value, dtype=DTYPE
    )
    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, groups, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, groups, dim_value, device=DEVICE, dtype=DTYPE)
    expected = _causal_softmax_ref(query, key, value)
    actual = module(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=RTOL_STRICT, atol=ATOL_STRICT)


def test_softmax_decode_forward_matches_reference():
    _seed()
    # single token decode
    batch, heads, seqlen, kv_len, dim, dim_value = 1, 2, 1, 256, 64, 64
    module = _build_softmax_decode_attention(
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
    module = _build_sigmoid_attention(batch, heads, seqlen, dim, dim_value)
    query = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    key = torch.randn(batch, seqlen, heads, dim, device=DEVICE, dtype=DTYPE)
    value = torch.randn(batch, seqlen, heads, dim_value, device=DEVICE, dtype=DTYPE)
    sigmoid_bias = torch.tensor([1.0], device=DEVICE, dtype=torch.float32).uniform_(
        -10.0, 2.0
    )
    expected = _sigmoid_ref(query, key, value, sigmoid_bias)
    actual = module(query, key, value, sigmoid_bias)
    torch.testing.assert_close(actual, expected, rtol=RTOL_LOOSE, atol=ATOL_LOOSE)


def test_relu_attention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 2, 128, 64, 64
    module = _build_relu_attention(batch, heads, seqlen, dim, dim_value, dtype=DTYPE)
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
    module = _build_sparse_gqa_decode_attention(
        batch, heads, groups, seqlen, dim, dim_value, BLOCK=block, dtype=DTYPE
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


def test_retention_parallel_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 8, 128, 64, 64
    module = _build_retention_parallel_attention(
        batch, heads, seqlen, dim, dim_value, dtype=DTYPE
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


def test_mla_decode_forward_matches_reference():
    _seed()
    batch, heads, kv_heads, seqlen, dim, dim_value = 1, 64, 1, 128, 128, 64
    module = _build_mla_decode_attention(
        batch, heads, seqlen, dim, dim_value, HK=kv_heads, HV=kv_heads, dtype=DTYPE
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
