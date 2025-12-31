# TL_IMPORT = """
import torch
import tilelang as tl
import tilelang.language as T
import torch.nn.functional as F

from math import floor

from autotuner.arch import AttnDevice, AttnDeviceAMD, H100

if torch.version.cuda is not None:
    AttnDeviceDict = AttnDevice
elif torch.version.hip is not None:
    AttnDeviceDict = AttnDeviceAMD
else:
    raise RuntimeError("Unsupported device type")
current_device = torch.cuda.current_device()
device_cap = torch.cuda.get_device_capability(current_device)
try:
    attn_device = AttnDeviceDict[device_cap]()
except KeyError:
    attn_device = H100()

# TL_GLOBAL_FUNC = """
def fast_tanh(A, B):
    return T.call_extern("handle", "fasttanh", T.address_of(A), T.address_of(B))

def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(
        dQ.shape, lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2]
    )

# TL_KERNEL = """
def kernel(batch, heads, seq_len, seq_len_kv, dim, dimv, 
           num_split=4,
        block_M = None, block_N = None, num_stages = None, thread_num = None,
        shared_fuse = None):
    # scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) # 0.69314718  loge(2)
    shape = [batch, seq_len, heads, dim]
    shape_k = [batch, seq_len_kv, heads, dim]
    shape_v = [batch, seq_len_kv, heads, dimv]
    shape_o = [batch, seq_len, heads, dimv]
    part_shape_o = [batch, seq_len, heads, num_split, dimv]
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    
    block_M2 = 128
    
    # TODO: mask
    is_casual = {{is_inf_mask}} # True
    # shared_fuse = True
    assert(is_casual == False)

    @T.macro
    {{score_mod_func_def | indent(4)}}
    
    @T.macro
    {{online_func_def | indent(8)}}

        
    @T.macro
    def main_split(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape_k, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        {{custom_fwd_inputs | indent(8)}}

        Output_partial: T.Buffer(part_shape_o, dtype), # type: ignore
        {{final_rowscales_output | indent(8)}}
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads*batch, num_split, threads=thread_num) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dimv], dtype)
            O_shared = T.alloc_shared([block_M, dimv], dtype)
            # V_local = T.alloc_fragment([block_N, dimv], dtype)
            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_shared = T.alloc_shared([block_M, block_N], accum_dtype)
            scores_1 = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_s_cast_1 = T.alloc_fragment([block_M, block_N], dtype)
            # acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

            {{custom_fwd_inputs_init | indent(12)}}
            
            mid = bx
            hid = by % heads
            bid = by // heads
            sid = bz

            T.annotate_layout({
                # Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                scores_shared: tl.layout.make_swizzled_layout(scores_shared),
                {{swizzle_shared | indent(16)}}
            })
            # automatic bound check?
            T.copy(Q[bid, mid * block_M : (mid + 1) * block_M, hid, :], Q_shared)
            {{custom_fwd_inputs_load_prolog | indent(12)}}
            T.fill(acc_o, 0)
            T.fill({{o_scale_varname}}, 1.0)

            {{online_rowscales_initvalue | indent(12)}}

            # TODO: mask
            loop_range = (
                T.ceildiv(seq_len_kv // num_split, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bid, (seq_len_kv // num_split) * sid + k * block_N : (seq_len_kv // num_split) * sid + (k + 1) * block_N, hid, :], K_shared)

                # TODO: copy custom_fwd_input_tensor in score_mod&online_func
                {{custom_fwd_inputs_load_shared | indent(16)}}

                # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
                T.clear(scores)
                
                T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy= (T.GemmWarpPolicy.FullRow if (not shared_fuse) else T.GemmWarpPolicy.FullCol))
                T.copy(V[bid, (seq_len_kv // num_split) * sid + k * block_N : (seq_len_kv // num_split) * sid + (k + 1) * block_N, hid, :], V_shared)
                    
                {{custom_fwd_inputs_load_s2r | indent(16)}}
                # call score_mod
                {{call_score_mod | indent(16)}}
                    
                # call online_func
                if shared_fuse:
                    T.copy(scores, scores_shared)
                    T.copy(scores_shared, scores_1)
                    {{call_online_func | indent(20)}}
                    T.copy(scores_1, acc_s_cast_1)

                else:
                    {{call_online_func | indent(20)}}
                    T.copy(scores, acc_s_cast)

                for i, j in T.Parallel(block_M, dimv):
                    acc_o[i, j] *= {{o_scale_varname}}[i]
                
                # update online_rowscales
                {{online_rowscales_update | indent(16)}}

                if shared_fuse:
                    T.gemm(acc_s_cast_1, V_shared, acc_o, policy=(T.GemmWarpPolicy.FullCol))
                else:
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            
            # online_fwd_epilogue
            {{online_func_epilogue | indent(12)}}

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, mid * block_M : (mid + 1) * block_M, hid, sid, :])

            # save final_rowscale
            {{final_rowscales_save | indent(12)}}
        
    @T.macro
    def combine(
        # g_lse
        {{final_rowscales_output | indent(8)}}
        Output_partial: T.Buffer(part_shape_o, dtype),
        Output: T.Buffer(shape_o, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M2), heads, batch, threads=128) as (bx, by, bz):
            po_local = T.alloc_fragment([block_M2, dim], dtype)
            po_shared = T.alloc_shared([block_M2, dim], dtype)
            o_accum_local = T.alloc_fragment([block_M2, dim], accum_dtype)
            o_shared = T.alloc_shared([block_M2, dim], dtype)
            
            # lse_local = T.alloc_fragment([num_split, block_M], dtype)
            # row_sum_local = T.alloc_fragment([num_split, block_M], dtype)
            # lse_logsum_local = T.alloc_fragment([block_M], accum_dtype)
            # lse_max_local = T.alloc_fragment([block_M], accum_dtype)
            # scale_o_local = T.alloc_fragment([num_split, block_M], accum_dtype)
            {{combinekernel_input_init | indent(12)}}
            
            T.annotate_layout(
                {
                    o_accum_local: T.Fragment(o_accum_local.shape, forward_thread_fn=lambda i, j: i),
                    o_shared: tl.layout.make_swizzled_layout(o_shared),
                    po_shared: tl.layout.make_swizzled_layout(po_shared),
                }
            )

            T.clear(o_accum_local)
            # T.copy(g_lse[bz, by, :, bx * block_M : (bx + 1) * block_M,], lse_local)
            {{combinekernel_input_load | indent(12)}}
            
            # T.reduce_max(lse_local, lse_max_local, dim=0, clear=False)
            # for i,j in T.Parallel(num_split, block_M):
            #     row_sum_local[i, j] = T.exp2(lse_local[i, j] - lse_max_local[j])
            # T.reduce_sum(row_sum_local, lse_logsum_local, dim=0)
            # for i in T.Parallel(block_M):
            #     lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
            # for i, j in T.Parallel(num_split, block_M):
            #     scale_o_local[i, j] = T.exp2(lse_local[i, j] - lse_logsum_local[j])
            {{combinekernel_combine | indent(12)}}
            
            for k in T.Pipelined(num_split, num_stages=2):
                T.copy(Output_partial[bz, bx * block_M2:(bx + 1) * block_M2, by, k, :], po_shared)
                T.copy(po_shared, po_local)
                for i, j in T.Parallel(block_M2, dim):
                    # o_accum_local[i, j] += po_local[i, j] * scale_o_local[k, i]
                    o_accum_local[i, j] += po_local[i, j] * {{combinekernel_scale_o_varname}}[k, i]
            T.copy(o_accum_local, o_shared)
            T.copy(o_shared, Output[bz, bx * block_M2 : (bx + 1) * block_M2, by, :])

    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape_k, dtype),
        V: T.Buffer(shape_v, dtype),
        {{custom_fwd_inputs | indent(8)}}
        # g_lse
        {{final_rowscales_output | indent(8)}}
        Output_partial: T.Buffer(part_shape_o, dtype), # [batch, seqlen_q, heads, num_split, dim]
        Output: T.Buffer(shape_o, dtype),
    ):
        # flash_attn_split(Q, K, V, glse, Output_partial)
        main_split(Q, K, V, {{custom_fwd_inputs_list}} Output_partial, {{final_rowscales_list}})
        combine({{final_rowscales_list}} Output_partial, Output)

    return main

# TODO: tune
tuned_config = {
    'block_M': {{block_M}},
    'block_N': {{block_N}},
    'num_stages': {{stages}} if attn_device.platform == "CUDA" else 0,
    'thread_num': {{thread_num}},
    'shared_fuse': {{shared_fuse}}
}    
program = kernel(
    {{BATCH}}, {{HEADS}}, {{SEQ_LEN}}, {{SEQ_LEN_KV}}, {{DIM}}, {{DIMV}},
    4, **tuned_config
)
mod = tl.compile(
    program,
    out_idx={{output_idx_list}},
    execution_backend="dlpack",
)
# TL_INFERFACE = """
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, *custom_fwd_inputs):
        BATCH, N_CTXQ, H, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        N_CTXKV = k.shape[1]
        assert(N_CTXQ <= {{block_M}})
        # TODO: pad
        N_CTXQOLD = N_CTXQ
        if N_CTXQ < {{block_M}}:
            q = F.pad(q, (0, 0, 0 , 0, 0, {{block_M}} - N_CTXQ))
            N_CTXQ = {{block_M}}
            
        global mod
        
        num_split = 4
        output_idx_list = {{output_idx_list}}
        O_partial = torch.empty(BATCH, N_CTXQ, H, num_split, D_HEADV, dtype=q.dtype, device=q.device)
        {{torch_alloc_final_rowscales | indent(8)}}
        
        if len(output_idx_list) == 1:
            o = mod(q, k, v, *custom_fwd_inputs, {{final_rowscales_list}} O_partial)
            final_scale = []
        else:
            o, *final_scale = mod(q, k, v, *custom_fwd_inputs, {{final_rowscales_list}} O_partial)

        if N_CTXQOLD < {{block_M}}:
            o = o[:, :N_CTXQOLD, :, :]
        return o
    
    @staticmethod
    def backward(ctx, do):
        pass

attention = _attention.apply


