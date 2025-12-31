import torch
import tilelang
import tilelang.language as T
import math

def flashattn(batch, heads, heads_kv, dim, dim_v):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // heads_kv

    def kernel_func(block_N, block_H, num_split, num_stages, threads, max_cache_seqlen, num_blocks):
        shape_q = [batch, heads, dim]
        shape_k = [batch, max_cache_seqlen, heads_kv, dim]
        shape_v = [batch, max_cache_seqlen, heads_kv, dim_v]
        shape_mask = [batch, heads_kv, num_blocks]
        shape_o = [batch, heads, dim_v]
        part_shape = [batch, heads, num_split, dim_v]
        valid_block_H = min(block_H, kv_group_num)

        @T.macro
        def flash_attn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                block_mask: T.Tensor(shape_mask, "bool"),
                cache_seqlens: T.Tensor([batch], "int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim_v], dtype)
                # O_shared = T.alloc_shared([valid_block_H, dim_v], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim_v], accum_dtype)

                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)
                has_valid_block = T.alloc_var("bool")

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                blocks_per_split = T.floordiv(num_blocks, num_split)
                remaining_blocks = T.floormod(num_blocks, num_split)
                loop_range = (blocks_per_split + T.if_then_else(sid < remaining_blocks, 1, 0))
                start = blocks_per_split * sid + T.min(sid, remaining_blocks)
                has_valid_block = False
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if block_mask[bid, hid, start + k]:
                        has_valid_block = True
                        T.copy(
                            K[bid, (start + k) * block_N:(start + k + 1) * block_N, cur_kv_head, :],
                            K_shared)
                        T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.if_then_else((start + k) * block_N + j
                                                         >= cache_seqlens[bx],
                                                         -T.infinity(accum_dtype), acc_s[i, j])
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_H):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale -
                                                     scores_max[i] * scale)
                        for i, j in T.Parallel(block_H, block_N):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_H):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)
                        for i, j in T.Parallel(block_H, dim_v):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(
                            V[bid, (start + k) * block_N:(start + k + 1) * block_N, cur_kv_head, :],
                            V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if has_valid_block:
                    for i, j in T.Parallel(block_H, dim_v):
                        acc_o[i, j] /= logsum[i]
                    for i in T.Parallel(block_H):
                        logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]

                for i, j in T.Parallel(block_H, dim_v):
                    if i < valid_block_H:
                        Output_partial[bid, hid * valid_block_H + i, sid, j] = acc_o[i, j]

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim_v], accum_dtype)
                o_accum_local = T.alloc_fragment([dim_v], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_local([1], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                lse_max_local[0] = -T.infinity(accum_dtype)
                for k in T.serial(num_split):
                    lse_max_local[0] = T.max(lse_max_local[0], glse[bz, by, k])
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim_v):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim_v):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim_v):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def main(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                block_mask: T.Tensor(shape_mask, "bool"),
                cache_seqlens: T.Tensor([batch], "int32"),
                glse: T.Tensor([batch, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, block_mask, cache_seqlens, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return main

    return kernel_func


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

class SparseFlashAttn(torch.nn.Module):

    def __init__(self, batch, heads, heads_kv, dim, dim_v, block_size):
        super(SparseFlashAttn, self).__init__()
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.dim_v = dim_v
        self.block_size = block_size

        self.block_H = 64

        program = flashattn(batch, heads, heads_kv, dim, dim_v)(
            block_N=block_size,
            block_H=self.block_H,
            num_split=T.symbolic("num_split"),
            num_stages=2,
            threads=128,
            max_cache_seqlen={{SEQ_LEN_KV}}, # T.symbolic("max_cache_seqlen"), # Tilelang0.1.5 bug
            num_blocks=T.symbolic("num_blocks"))

        self.kernel = tilelang.compile(
            program, out_idx=-1, execution_backend="cython")

        props = torch.cuda.get_device_properties(torch.device("cuda:0"))
        self.num_sm = props.multi_processor_count

    def forward(self, query, key, value, block_mask, cache_seqlens):
        query = query.squeeze(1)  # [B, 1, H, D] -> [B, H, D]
        batch = self.batch
        heads = self.heads
        heads_kv = self.heads_kv
        dim_v = self.dim_v
        dim = self.dim
        block_size = self.block_size
        block_H = self.block_H
        max_cache_seqlen = key.shape[1]
        # get num_split
        max_selected_blocks = (max_cache_seqlen + block_size - 1) // block_size
        num_m_blocks = 1 * (heads // heads_kv + block_H - 1) // block_H
        num_n_blocks = max_selected_blocks

        size_one_kv_head = max_selected_blocks * block_size * (
            dim + dim_v) * 2  #kv_seqlen * (dim + dim_v) * 2
        total_mblocks = batch * heads_kv * num_m_blocks
        # num_sm = 132
        num_sm = self.num_sm
        num_split = num_splits_heuristic(
            total_mblocks,
            num_sm,
            num_n_blocks,
            num_m_blocks,
            size_one_kv_head,
            is_causal_or_local=True,
            max_splits=128)
        # print("num_split: ", num_split)
        glse = torch.empty((batch, heads, num_split), dtype=torch.float32, device='cuda')
        Output_partial = torch.empty((batch, heads, num_split, dim_v),
                                     dtype=torch.float32,
                                     device='cuda')
        output = self.kernel(query, key, value, block_mask, cache_seqlens, glse, Output_partial)
        return output

attention = SparseFlashAttn(
    {{BATCH}}, {{HEADS}}, {{GROUPS}}, {{DIM}}, {{DIMV}}, {{infer_mask_block_N}}
)
