import math

import pytest
import torch
import torch.nn.functional as F
from attn_engine import AttentionEngine, LinearAttentionEngine, OnlineFunc
from benchmark.bench_utils import assert_close
from core import CustomIO, SymbolScalar, Var, meta_tensor
from einops import einsum, rearrange, repeat

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
        row_sum = (lse - lse_max).exp2()
        row_sum_sum = row_sum.get_reduce("sum")
        lse_sum = row_sum_sum.log2() + lse_max
        return (lse - lse_sum).exp2()

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


def _build_gated_retention_attention(B, H, S, D, DV, dtype=LINEAR_DTYPE):
    scale = 1 / D**0.5

    def q_mod(q, custom_io):
        return q * scale

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    return LinearAttentionEngine(
        qkv_meta, q_mod=q_mod, custom_io=CustomIO({}), tune=False
    )


def _build_retnet_recurrent_attention(B, H, S, D, DV, dtype=LINEAR_DTYPE):
    scale = 1 / D**0.5

    def decay_mod(decay, custom_io):
        return decay.log()

    def q_mod(q, custom_io):
        return q * scale

    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    return LinearAttentionEngine(
        qkv_meta,
        q_mod=q_mod,
        decay_mod=decay_mod,
        custom_io=CustomIO({}),
        tune=False,
    )


def _build_mamba2_attention(B, HQ, S, D, DV, HK, HV, dtype=LINEAR_DTYPE):
    def decay_mod(decay, custom_io):
        return decay * custom_io.input_tensors["A"]

    def v_mod(v, custom_io):
        return v * custom_io.input_tensors["dt"]

    qkv_meta = (
        meta_tensor(B, HQ, S, D, dtype=dtype),
        meta_tensor(B, HK, S, D, dtype=dtype),
        meta_tensor(B, HV, S, DV, dtype=dtype),
    )
    custom_io = CustomIO(
        {
            "A": (1, "heads"),
            "dt": ("batch", "heads", "seq_len"),
        }
    )
    return LinearAttentionEngine(
        qkv_meta,
        decay_mod=decay_mod,
        v_mod=v_mod,
        custom_io=custom_io,
        tune=False,
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


def _naive_chunk_simple_gla(query, key, value, gate, chunk_size=64, scale=None):
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


def _retnet_recurrent_ref(query, key, value):
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


def _mamba2_ref(value, delta_t, a_param, key, query, chunk_size=64):
    def chunk_state_ref(b_tensor, x_tensor, dt_tensor, d_a_cumsum):
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

    def state_passing_ref(states, d_a_chunk_cumsum):
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
        b_tensor, c_tensor, x_tensor, dt_tensor, d_a_cumsum, prev_states
    ):
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
    batch, heads, seqlen, kv_len, dim, dim_value = 1, 2, 128, 256, 64, 64
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


def test_gated_retention_forward_matches_reference():
    _seed()
    batch, heads, seqlen, dim, dim_value = 1, 4, 128, 64, 64
    module = _build_gated_retention_attention(
        batch, heads, seqlen, dim, dim_value, dtype=LINEAR_DTYPE
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
    batch, heads, seqlen, dim, dim_value = 1, 4, 128, 64, 64
    module = _build_retnet_recurrent_attention(
        batch, heads, seqlen, dim, dim_value, dtype=LINEAR_DTYPE
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
        1,
        128,
        64,
        64,
    )
    module = _build_mamba2_attention(
        batch,
        query_heads,
        seqlen,
        dim,
        dim_value,
        HK=key_heads,
        HV=value_heads,
        dtype=LINEAR_DTYPE,
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
