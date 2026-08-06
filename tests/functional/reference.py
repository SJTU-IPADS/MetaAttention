from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


def causal_softmax(query, key, value, softmax_scale: float | None = None):
    dim = query.shape[-1]
    groups = query.shape[2] // key.shape[2]
    if softmax_scale is None:
        softmax_scale = 1 / dim**0.5
    grouped_query = rearrange(query, "b s (h g) d -> b s g h d", g=groups)
    scores = einsum(grouped_query, key, "b s g h d, b t h d -> b g h s t")
    mask = torch.tril(
        torch.ones(query.shape[1], key.shape[1], device=scores.device, dtype=torch.bool)
    )[None, None]
    scores = scores.masked_fill(~mask, float("-inf"))
    attention = F.softmax(scores * softmax_scale, dim=-1)
    output = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    return rearrange(output, "b g h s d -> b s (h g) d")


def decode_softmax(query, key, value, softmax_scale: float | None = None):
    dim = query.shape[-1]
    groups = query.shape[2] // key.shape[2]
    if softmax_scale is None:
        softmax_scale = 1 / dim**0.5
    grouped_query = rearrange(query, "b s (h g) d -> b s g h d", g=groups)
    scores = einsum(grouped_query, key, "b s g h d, b t h d -> b g h s t")
    attention = F.softmax(scores * softmax_scale, dim=-1)
    output = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    return rearrange(output, "b g h s d -> b s (h g) d")


def sigmoid_attention(query, key, value, sigmoid_bias):
    groups = query.shape[2] // key.shape[2]
    grouped_query = rearrange(query, "b s (h g) d -> b s g h d", g=groups)
    scores = einsum(grouped_query, key, "b s g h d, b t h d -> b g h s t")
    mask = torch.tril(
        torch.ones(query.shape[1], key.shape[1], device=scores.device, dtype=torch.bool)
    )[None, None]
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = scores + sigmoid_bias.to(scores.dtype)
    attention = (torch.tanh(scores * 0.5) + 1) * 0.5
    output = einsum(attention, value, "b g h s t, b t h d -> b g h s d")
    return rearrange(output, "b g h s d -> b s (h g) d")


def relu_attention(query, key, value):
    dim = query.shape[-1]
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / dim**0.5
    scores = F.relu(scores)
    return torch.einsum("bhqk,bkhd->bqhd", scores, value)


def sparse_gqa_decode(query, key, value, block_mask, cache_seqlens, block_size):
    query = query.squeeze(1)
    batch, heads, dim = query.shape
    heads_kv = key.shape[2]
    num_blocks = block_mask.shape[-1]
    groups = heads // heads_kv
    key = rearrange(key, "b n h d -> b h n d")
    value = rearrange(value, "b n h d -> b h n d")
    query = rearrange(query, "b (h g) d -> b g h d", g=groups)
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
    positions = torch.arange(scores.shape[-1], device=query.device)[None]
    pad_mask = positions >= cache_seqlens[:, None]
    scores = scores.masked_fill(pad_mask[:, None, None], float("-inf"))
    attention = F.softmax(scores / dim**0.5, dim=-1)
    output = einsum(attention, value, "b g h s, b h s d -> b g h d")
    return rearrange(output, "b g h d -> b (h g) d")


def mla_decode(query, query_pe, key_value, key_pe):
    query = query.squeeze(1)
    query_pe = query_pe.squeeze(1)
    dim = query.shape[-1]
    pe_dim = query_pe.shape[-1]
    groups = query.shape[1] // key_value.shape[2]
    query = rearrange(query, "b (h g) d -> b g h d", g=groups)
    query_pe = rearrange(query_pe, "b (h g) d -> b g h d", g=groups)
    key_value = rearrange(key_value, "b n h d -> b h n d")
    key_pe = rearrange(key_pe, "b n h d -> b h n d")
    full_query = torch.cat([query, query_pe], dim=-1)
    full_key = torch.cat([key_value, key_pe], dim=-1)
    scores = einsum(full_query, full_key, "b g h d, b h s d -> b g h s")
    attention = F.softmax(scores / (dim + pe_dim) ** 0.5, dim=-1)
    output = einsum(attention, key_value, "b g h s, b h s d -> b g h d")
    return rearrange(output, "b g h d -> b (h g) d").unsqueeze(1)


def retention_parallel(query, key, value, mask):
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
    masked_scores = scores * mask
    row_sum = masked_scores.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.einsum("bhqk,bkhd->bqhd", masked_scores / row_sum, value)


def gated_retention(query, key, value, gate, chunk_size: int = 64):
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    gate = gate.to(torch.float32)
    scale = 1.0 / query.shape[-1] ** 0.5
    total_steps = query.shape[-2]
    pad_len = (chunk_size - total_steps % chunk_size) % chunk_size
    if pad_len:
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
        attention = (
            query_chunk @ key_chunk.transpose(-1, -2) * local_mask[:, :, chunk_idx]
        )
        carry = (query_chunk * decay[:, :, chunk_idx, :, None].exp()) @ state
        output[:, :, chunk_idx] = carry + attention @ value_chunk
        end_decay = decay[:, :, chunk_idx, -1, None]
        key_scale = (end_decay - decay[:, :, chunk_idx]).exp()[..., None]
        state = (
            state * end_decay.exp()[..., None]
            + (key_chunk * key_scale).transpose(-1, -2) @ value_chunk
        )
    return rearrange(output, "b h n c d -> b h (n c) d")[:, :, :total_steps]


def retnet_recurrent(query, key, value):
    original_dtype = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    _, heads, seq_len, dim = query.shape
    slope = (
        1
        - query.new_tensor(2.0).pow(
            -5.0 - query.new_tensor(range(heads), dtype=torch.float)
        )
    ).log2()
    positions = query.new_tensor(range(seq_len), dtype=torch.float)
    decay = torch.exp2((positions[:, None] - positions[None, :]) * slope.view(-1, 1, 1))
    decay = decay * positions[:, None].ge(positions[None, :])
    scores = torch.einsum(
        "bhqd,bhkd,hqk->bhqk", query * dim**-0.5, key, decay.to(query.dtype)
    )
    return torch.einsum("bhqk,bhkd->bhqd", scores, value).to(original_dtype)


def mamba2(value, delta_t, a_param, key, query, chunk_size: int = 64):
    def chunk_state(b_tensor, x_tensor, dt_tensor, d_a_cumsum):
        batch, seqlen, nheads, headdim = x_tensor.shape
        _, _, nchunks, local_chunk = dt_tensor.shape
        assert seqlen == nchunks * local_chunk
        assert nheads % b_tensor.shape[2] == 0
        expanded_b = repeat(
            b_tensor, "b l g d -> b l (g h) d", h=nheads // b_tensor.shape[2]
        )
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

    def state_passing(states, d_a_chunk_cumsum):
        initial_states = torch.zeros_like(states[:, 0])
        states = torch.cat(
            [rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1
        )
        d_a_chunk_cumsum = F.pad(d_a_chunk_cumsum, (1, 0))
        d_a_chunk_cumsum = torch.cumsum(d_a_chunk_cumsum, dim=-1)
        segment_sum = d_a_chunk_cumsum[:, :, :, None] - d_a_chunk_cumsum[:, :, None, :]
        decay_chunk = torch.exp(segment_sum)
        causal_mask = torch.tril(
            torch.ones(
                decay_chunk.shape[-1],
                decay_chunk.shape[-1],
                device=states.device,
                dtype=torch.bool,
            )
        )
        decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
        output = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(states.dtype), states)
        return output[:, :-1]

    def chunk_scan(b_tensor, c_tensor, x_tensor, dt_tensor, d_a_cumsum, previous):
        batch, seqlen, nheads, _ = x_tensor.shape
        _, _, groups, dstate = b_tensor.shape
        _, _, nchunks, local_chunk = dt_tensor.shape
        assert seqlen == nchunks * local_chunk
        expanded_b = repeat(b_tensor, "b l g d -> b l (g h) d", h=nheads // groups)
        expanded_c = repeat(c_tensor, "b l g d -> b l (g h) d", h=nheads // groups)
        cb = torch.einsum(
            "bclhn,bcshn->bchls",
            rearrange(expanded_c, "b (c l) h n -> b c l h n", c=nchunks),
            rearrange(expanded_b, "b (c s) h n -> b c s h n", c=nchunks),
        )
        segment_sum = d_a_cumsum[:, :, :, :, None] - d_a_cumsum[:, :, :, None, :]
        scores_decay = cb * rearrange(segment_sum.exp(), "b h c l s -> b c h l s")
        causal_mask = torch.tril(
            torch.ones(
                local_chunk, local_chunk, device=x_tensor.device, dtype=torch.bool
            )
        )
        scores_decay = scores_decay.masked_fill(~causal_mask, 0)
        output = torch.einsum(
            "bchls,bhcs,bcshp->bclhp",
            scores_decay.to(x_tensor.dtype),
            dt_tensor.to(x_tensor.dtype),
            rearrange(x_tensor, "b (c s) h p -> b c s h p", c=nchunks),
        )
        state_decay = torch.exp(rearrange(d_a_cumsum, "b h c l -> b c l h 1"))
        output = (
            output
            + torch.einsum(
                "bclhn,bchpn->bclhp",
                rearrange(expanded_c, "b (c l) h n -> b c l h n", c=nchunks),
                previous.to(expanded_c.dtype),
            )
            * state_decay
        )
        return rearrange(output, "b c l h p -> b (c l) h p")

    dt_chunks = rearrange(delta_t, "b (c l) h -> b h c l", l=chunk_size).float()
    d_a = dt_chunks * rearrange(a_param, "h -> h 1 1")
    d_a_cumsum = torch.cumsum(d_a, dim=-1)
    states = chunk_state(key, value, dt_chunks, d_a_cumsum)
    state_dtype = states.dtype
    if state_dtype not in (torch.float32, torch.float64):
        states = states.to(torch.float32)
    states = rearrange(
        state_passing(
            rearrange(states, "... p n -> ... (p n)"),
            d_a_cumsum[:, :, :, -1],
        ),
        "... (p n) -> ... p n",
        n=key.shape[-1],
    ).to(state_dtype)
    return chunk_scan(key, query, value, dt_chunks, d_a_cumsum, states)
