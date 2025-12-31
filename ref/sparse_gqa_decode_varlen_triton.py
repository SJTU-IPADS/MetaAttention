import torch
import triton
import triton.language as tl
import argparse
from einops import rearrange, einsum
import torch.nn.functional as F

import math
import time


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, num_m_blocks, size_one_kv_head,
                         is_causal_or_local, max_splits):
    """
    Determines the optimal number of splits for maximizing GPU occupancy while balancing memory efficiency.

    Parameters:
    - total_mblocks (int): Total number of m_blocks.
    - num_SMs (int): Number of Streaming Multiprocessors (SMs) in the GPU.
    - num_n_blocks (int): Number of n_blocks.
    - num_m_blocks (int): Number of m_blocks.
    - size_one_kv_head (int): Size of one KV head in bytes.
    - is_causal_or_local (bool): Indicates whether the operation is causal or local.
    - max_splits (int): Maximum number of allowed splits.

    Returns:
    - int: The optimal number of splits.
    """
    # If we have enough m_blocks to almost fill the SMs, prefer 1 split unless memory constraints apply.
    if total_mblocks >= 0.8 * num_SMs:
        size_l2 = 50 * 1024 * 1024  # L2 cache size assumption (50MB)
        # Only split if each KV head is too large for L2 and there are enough m_blocks
        if size_one_kv_head > size_l2 and num_m_blocks >= num_SMs * 2 and not is_causal_or_local:
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        else:
            return 1

    # If num_n_blocks is too small, we don't split
    if num_n_blocks <= 4:
        return 1

    # Limit max_splits to a reasonable range
    max_splits = min(max_splits, num_SMs, num_n_blocks)

    max_efficiency = 0.0
    efficiency = []

    # Compute efficiency for different splits
    for num_splits in range(1, max_splits + 1):
        n_waves = (total_mblocks * num_splits) / num_SMs
        eff = n_waves / math.ceil(n_waves)
        # Track max efficiency
        if eff > max_efficiency:
            max_efficiency = eff

        efficiency.append(eff)

    # Find the smallest number of splits that achieves at least 85% of max efficiency
    for num_splits in range(1, max_splits + 1):
        if efficiency[num_splits - 1] >= 0.85 * max_efficiency:
            return num_splits

    return 1


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4]\
        for num_stages in [1, 2, 3, 4, 7]
    ],
    key=['BLOCK_H', 'BLOCK_N', 'BLOCK_D'],
)
@triton.jit
def _split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_seqlens_ptr,
    o_partial_ptr,
    lse_partial_ptr,
    mask_ptr,
    sm_scale,
    num_splits,
    gqa_group_size,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_o_b,
    stride_o_h,
    stride_o_split,
    stride_o_d,
    stride_lse_b,
    stride_lse_h,
    stride_lse_split,
    stride_mask_b,
    stride_mask_h,
    stride_mask_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx_kv = tl.program_id(1)
    split_idx = tl.program_id(2)

    head_idx_q = head_idx_kv * gqa_group_size
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    cache_seqlens = tl.load(cache_seqlens_ptr + batch_idx)
    num_blocks = (cache_seqlens + BLOCK_N - 1) // BLOCK_N
    blocks_per_split = tl.floor(num_blocks / num_splits).to(tl.int32)
    remaining_blocks = num_blocks % num_splits
    if split_idx < remaining_blocks:
        loop_range = blocks_per_split + 1
    else:
        loop_range = blocks_per_split

    q_ptr += batch_idx * stride_q_b + head_idx_q * stride_q_h
    k_cache_ptr += batch_idx * stride_k_b + head_idx_kv * stride_k_h + offs_n[
        None, :] * stride_k_s + offs_d[:, None] * stride_k_d
    v_cache_ptr += batch_idx * stride_v_b + head_idx_kv * stride_v_h + offs_n[:,
                                                                              None] * stride_v_s + offs_d[
                                                                                  None, :] * stride_v_d
    mask_ptr += batch_idx * stride_mask_b + head_idx_kv * stride_mask_h

    q = tl.load(
        q_ptr + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=offs_h[:, None] < gqa_group_size)
    start = blocks_per_split * split_idx + tl.minimum(split_idx, remaining_blocks)
    for block_idx in range(loop_range):
        start_n = (start + block_idx) * BLOCK_N
        mask_val = tl.load(mask_ptr + (start + block_idx) * stride_mask_s)
        if mask_val == 1:
            k_ptr = k_cache_ptr + start_n * stride_k_s
            v_ptr = v_cache_ptr + start_n * stride_v_s

            k = tl.load(k_ptr, mask=start_n + offs_n[None, :] < cache_seqlens, other=0.0)
            v = tl.load(v_ptr, mask=start_n + offs_n[:, None] < cache_seqlens, other=0.0)

            qk = tl.dot(q, k)
            qk = qk * sm_scale
            qk = tl.where(start_n + offs_n[None, :] < cache_seqlens, qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p = p.to(v.type.element_ty)
            acc += tl.dot(p, v)
            m_i = m_ij

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(o_partial_ptr.dtype.element_ty)

    lse_partial_ptr += batch_idx * stride_lse_b + (
        head_idx_q + offs_h) * stride_lse_h + split_idx * stride_lse_split
    tl.store(lse_partial_ptr, m_i, mask=offs_h < gqa_group_size)

    o_partial_ptr += batch_idx * stride_o_b + (
        head_idx_q +
        offs_h[:, None]) * stride_o_h + split_idx * stride_o_split + offs_d[None, :] * stride_o_d
    tl.store(o_partial_ptr, acc, mask=offs_h[:, None] < gqa_group_size)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4]\
        for num_stages in [1, 2, 3, 4, 7]
    ],
    key=['BLOCK_D'],
)
@triton.jit
def _merge_kernel(
    o_partial_ptr,
    lse_partial_ptr,
    o_ptr,
    lse_partial_stride_b,
    lse_partial_stride_h,
    lse_partial_stride_split,
    o_partial_stride_b,
    o_partial_stride_h,
    o_partial_stride_split,
    o_partial_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_d,
    BLOCK_D: tl.constexpr,
    num_splits: tl.constexpr,
    num_splits_pow2: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs_splits = tl.arange(0, num_splits_pow2)
    offs_d = tl.arange(0, BLOCK_D)

    lse_offsets = lse_partial_ptr + batch_idx * lse_partial_stride_b + head_idx * lse_partial_stride_h
    lse = tl.load(
        lse_offsets + offs_splits * lse_partial_stride_split,
        mask=offs_splits < num_splits,
        other=float("-inf"))

    lse_max = tl.max(lse)

    o_offsets = o_partial_ptr + batch_idx * o_partial_stride_b + head_idx * o_partial_stride_h
    o_partial = tl.load(
        o_offsets + offs_splits[:, None] * o_partial_stride_split +
        offs_d[None, :] * o_partial_stride_d,
        mask=offs_splits[:, None] < num_splits)
    sumexp_normalized_splitk = tl.exp(lse - lse_max)
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)
    numerator_normalized = tl.sum(o_partial * sumexp_normalized_splitk[:, None], axis=0)
    acc = numerator_normalized / sumexp_normalized
    acc = acc.to(o_ptr.dtype.element_ty)
    o_ptr += batch_idx * o_stride_b + head_idx * o_stride_h
    tl.store(o_ptr + offs_d * o_stride_d, acc)


def block_sparse_flash_decode_gqa_mask_triton(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    max_cache_seqlen,
    block_mask,
    block_size,
    sm_scale=None,
):
    q = q.squeeze(1)  # B, H, D
    batch, heads, dim = q.shape

    if sm_scale is None:
        sm_scale = 1 / math.sqrt(dim)

    _, max_cache_seqlen_cache, heads_kv, dim_v = v_cache.shape
    assert max_cache_seqlen == max_cache_seqlen_cache, "max_cache_seqlen mismatch"
    group_size = heads // heads_kv

    block_H = 16

    max_selected_blocks = (max_cache_seqlen + block_size - 1) // block_size
    num_m_blocks = 1 * (heads // heads_kv + block_H - 1) // block_H
    num_n_blocks = max_selected_blocks

    size_one_kv_head = max_selected_blocks * block_size * (
        dim + dim_v) * 2  #kv_seqlen * (dim + dim_v) * 2
    total_mblocks = batch * heads_kv * num_m_blocks
    num_sm = 64
    # num_sm = self.num_sm
    num_splits = num_splits_heuristic(
        total_mblocks,
        num_sm,
        num_n_blocks,
        num_m_blocks,
        size_one_kv_head,
        is_causal_or_local=True,
        max_splits=128)

    # print("num_splits:", num_splits, "num_blocks:", num_n_blocks)

    num_splits_pow2 = triton.next_power_of_2(num_splits)

    o_partial = torch.empty((batch, heads, num_splits, dim_v), device=q.device, dtype=q.dtype)
    lse_partial = torch.empty((batch, heads, num_splits), device=q.device, dtype=torch.float32)

    BLOCK_D = dim
    BLOCK_H = group_size if group_size > 16 else 16
    grid = (batch, heads_kv, num_splits)
    _split_kernel[grid](
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        o_partial,
        lse_partial,
        block_mask,
        sm_scale,
        num_splits,
        group_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_mask.stride(0),
        block_mask.stride(1),
        block_mask.stride(2),
        BLOCK_H=BLOCK_H,
        BLOCK_N=block_size,
        BLOCK_D=BLOCK_D,
    )

    output = torch.zeros((batch, heads, dim_v), device=q.device, dtype=q.dtype)
    grid = (batch, heads)
    _merge_kernel[grid](
        o_partial,
        lse_partial,
        output,
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_D=dim_v,
        num_splits=num_splits,
        num_splits_pow2=num_splits_pow2,
    )

    return output


