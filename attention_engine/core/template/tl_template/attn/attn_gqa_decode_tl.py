import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
from tilelang.profiler import cached

# TL_GLOBAL_FUNC = """
def fast_tanh(A, B):
    return T.call_extern("handle", "fasttanh", T.address_of(A), T.address_of(B))

def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2]
    )

def get_configs():
    block_N = [64, 128]
    block_H = [64]
    num_split = [2, 4, 8]
    num_stages = [1, 2, 3]
    threads = [128]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

    configs = [{
        'block_N': c[0],
        'block_H': c[1],
        'num_split': c[2],
        'num_stages': c[3],
        'threads': c[4]
    } for c in _configs]
    return configs


def kernel(batch, heads, groups, seqlen_kv, dim, dimv, tune=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape_q = [batch, 1, heads, dim]
    shape_k = [batch, seqlen_kv, groups, dim]
    shape_v = [batch, seqlen_kv, groups, dimv]
    shape_o = [batch, 1, heads, dimv]
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    kv_group_num = heads // groups

    def kernel_func(block_N, block_H, num_split, num_stages, threads):
        part_shape = [batch, heads, num_split, dimv]
        valid_block_H = min(block_H, kv_group_num)
        
         # TL_MAIN = """
         # TODO
        # @T.macro
        # def score_mod(
        #     # scores: T.Buffer([block_M, block_N], accum_dtype),
        #     {#{score_mod_inputs | indent(12)}#}
        #     ):
        #     {#{score_mod_body | indent(12)}#}
        #     pass
        
        # @T.macro
        # def online_func(
        #     # scores: T.Buffer([block_M, block_N], accum_dtype),
        #     {#{online_func_inputs | indent(12)}#}
        # ):
        #     {#{online_func_body | indent(12)}#}
        #     pass

        @T.macro
        def flash_attn(
                Q: T.Buffer(shape_q, dtype),
                K: T.Buffer(shape_k, dtype),
                V: T.Buffer(shape_v, dtype),
                {{custom_fwd_inputs | indent(8)}}
                
                mask: T.Buffer([batch, groups, 1, seqlen_kv], "uint8"),
                Output: T.Buffer([batch, heads, dimv], dtype),
                # {#{final_rowscales_output | indent(8)}#}
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dimv], dtype)
                O_shared = T.alloc_shared([valid_block_H, dimv], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dimv], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, 0, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_shared)
                    T.copy(mask[bid, cur_kv_head, 0, k * block_N:(k + 1) * block_N], mask_local)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                     -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(V[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, 0, hid * valid_block_H:(hid + 1) * valid_block_H, :])

        @T.macro
        def flash_attn_split(
                Q: T.Buffer(shape_q, dtype),
                K: T.Buffer(shape_k, dtype),
                V: T.Buffer(shape_v, dtype),
                mask: T.Buffer([batch, groups, 1, seqlen_kv], "uint8"),
                glse: T.Buffer([batch, heads, num_split], dtype),
                Output_partial: T.Buffer(part_shape, dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                mask_local = T.alloc_fragment([block_N], "uint8")
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                sid = bz
                cur_kv_head = hid // (kv_group_num // valid_block_H)

                T.copy(Q[bid, 0, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(
                        K[bid, (seqlen_kv // num_split) * sid +
                          k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                          cur_kv_head, :], K_shared)
                    T.copy(
                        mask[bid, cur_kv_head, 0, (seqlen_kv // num_split) * sid +
                             k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N], mask_local)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(mask_local[j] != 0, acc_s[i, j],
                                                     -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[bid, (seqlen_kv // num_split) * sid +
                          k * block_N:(seqlen_kv // num_split) * sid + (k + 1) * block_N,
                          cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

                T.copy(logsum[:valid_block_H],
                       glse[bid, hid * valid_block_H:(hid + 1) * valid_block_H, sid])
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local = T.alloc_fragment([num_split, 128], dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_fragment([128], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local:
                        T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                    lse_max_local:
                        T.Fragment(lse_max_local.shape, forward_thread_fn=lambda i: i),
                    # lse_local: (local_id, thread_id)
                    lse_local:
                        T.Fragment(lse_local.shape, forward_fn=lambda i, j: (j, i)),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                for k, j in T.Parallel(num_split, 128):
                    lse_local[k, j] = glse[bz, by, k]
                T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, by, k, i]
                    lse_local_split[0] = glse[bz, by, k]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum_local[i]

        @T.prim_func
        def main_split(
                Q: T.Buffer(shape_q, dtype),
                K: T.Buffer(shape_k, dtype),
                V: T.Buffer(shape_v, dtype),
                mask: T.Buffer([batch, groups, 1, seqlen_kv], "uint8"),
                glse: T.Buffer([batch, heads, num_split], dtype),
                Output_partial: T.Buffer(part_shape, dtype),
                Output: T.Buffer(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, mask, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def main_no_split(
                Q: T.Buffer(shape_q, dtype),
                K: T.Buffer(shape_k, dtype),
                V: T.Buffer(shape_v, dtype),
                mask: T.Buffer([batch, groups, 1, seqlen_kv], "uint8"),
                glse: T.Buffer([batch, heads, num_split], dtype),
                Output_partial: T.Buffer(part_shape, dtype),
                Output: T.Buffer(shape_o, dtype),
        ):
            flash_attn(Q, K, V, mask, Output)

        if num_split > 1:
            return main_split
        else:
            return main_no_split

    if tune:

        @autotune(
            configs=get_configs(),
            keys=["block_N", "block_H", "num_split", "num_stages", "threads"],
            warmup=10,
            rep=10)
        @jit(
            out_idx=[6],
            supply_type=tilelang.TensorSupplyType.Auto,
            ref_prog=ref_program,
            max_mismatched_ratio=0.05,
            profiler="auto")
        def kernel(block_N=None, block_H=None, num_split=None, num_stages=None, threads=None):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

        return kernel()
    else:

        def kernel(block_N, block_H, num_split, num_stages, threads):
            return kernel_func(block_N, block_H, num_split, num_stages, threads)

        return kernel

# def attention(
#     q,
#     k,
#     v,
#     *custom_fwd_inputs,
# ):
#     BATCH, N_CTXQ, H, D_HEAD = q.shape
#     _, N_CTXKV, G, D_HEADV = v.shape
#     program = kernel(BATCH, H,G, N_CTXKV,D_HEAD,D_HEADV)
#     mod = cached(program, [6], {{block_N}}, {{block_M}}, 8, 2, 128)
#     num_split = 8
#     {{torch_alloc_final_rowscales | indent(8)}}
#     O_partial = torch.empty(BATCH, H, num_split, D_HEADV, dtype=q.dtype, device=q.device)
#     o = mod(q, k, v, *custom_fwd_inputs, {{final_rowscales_list}} O_partial)
#     return o

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, *custom_fwd_inputs):
        BATCH, N_CTXQ, H, D_HEAD = q.shape
        _, N_CTXKV, G, D_HEADV = v.shape
        program = kernel(BATCH, H, G, N_CTXKV, D_HEAD, D_HEADV)
        mod = cached(program, [6], {{block_N}}, {{block_M}}, 8, 2, 128)
        num_split = 8
        {{torch_alloc_final_rowscales | indent(8)}}
        O_partial = torch.empty(BATCH, H, num_split, D_HEADV, dtype=q.dtype, device=q.device)
        o = mod(q, k, v, *custom_fwd_inputs, {{final_rowscales_list}} O_partial)
        return o

    @staticmethod
    def backward(ctx, grad_o):
        raise NotImplementedError("Backward not implemented for attention")

attention = _attention.apply
