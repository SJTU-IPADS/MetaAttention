# TL_IMPORT = """
import torch
import tilelang as tl
import tilelang
import tilelang.language as T
from tilelang.autotuner import *

import itertools
from typing import Tuple

import operator
import json
from functools import partial

from autotuner.arch import AttnDevice, H100
current_device = torch.cuda.current_device()
device_cap = torch.cuda.get_device_capability(current_device)
try:
    attn_device = AttnDevice[device_cap]()
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


def get_configs(DHEAD, DHEADV, device=H100()):
    
    # scheduling space
    block_M = [64, 128, 256]
    block_N = [32, 64, 128, 256]
    num_stages = [1, 2]
    thread_num = [128, 256]
    shared_fuse = [{{shared_fuse}},]# [True, False]
    
    # device config
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_N = device.mma_primitive[1]# 16
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    
    _configs = list(itertools.product(block_M, block_N, num_stages, thread_num, shared_fuse))
    
    configs = []
    # Inter tile
    block_M = [bM for bM in block_M if bM % MMA_ATOM_M == 0]
    block_N = [bN for bN in block_N if bN % MMA_ATOM_N == 0]
    thread_num = [t_n for t_n in thread_num if t_n % MMA_ATOM_TRHEADS == 0]
    for bM, bN, t_n in itertools.product(block_M, block_N, thread_num):
        plans = []
        # intra tile
        shared_fuse.sort()
        for s_fuse in shared_fuse:
            for n_s in num_stages:
                sharedmem = 2 * (bM * DHEAD + bN * DHEAD + bN * DHEADV) * n_s
                if s_fuse:
                    sharedmem += 2 * (bM * bN)
                reg_mem = 4 * (bM * DHEADV) + 4 * (bM * bN)
                conditions = [
                    sharedmem <= smem_cap,
                    reg_mem <= reg_cap and reg_mem <= reg_cap_per_thread * t_n,
                ]
                if all(conditions):
                    plans.append({
                        'block_M': bM,
                        'block_N': bN,
                        'num_stages': n_s,
                        'thread_num': t_n,
                        'shared_fuse': s_fuse
                    })
            if len(plans) > 0:
                break
        configs.extend(plans)
                    
    return configs
 
# TL_KERNEL = """
def kernel(batch, heads, seq_len, dim, dimv, 
           downsample_len_q, downsample_len_kv, tune=False):

    # scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) # 0.69314718  loge(2)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    # TODO: seqlenkv
    seq_len_kv = seq_len
    block_mask_shape = [batch, heads, downsample_len_q, downsample_len_kv]
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    block_mask_dtype = "int8"
    is_casual = {{is_casual}} # True
    
    def kernel_func(
        block_M = None, block_N = None, num_stages = None, thread_num = None,
        shared_fuse = None):

        
        # shared_fuse = True
        assert(downsample_len_kv <= seq_len_kv // block_N)
        assert(downsample_len_q <= seq_len // block_M)
        block_M_ratio = seq_len // block_M // downsample_len_q
        block_N_ratio = seq_len_kv // block_N // downsample_len_kv

        # TL_MAIN = """
        @T.macro
        {{score_mod_func_def | indent(8)}}
        
        @T.macro
        {{online_func_def | indent(8)}}

            
        @T.prim_func
        def main(
            Q: T.Buffer(shape, dtype), # type: ignore
            K: T.Buffer(shape, dtype), # type: ignore
            V: T.Buffer(shape_v, dtype), # type: ignore
            {{custom_fwd_inputs | indent(12)}}

            BlockSparseMask: T.Buffer(block_mask_shape, block_mask_dtype), # type: ignore
            Output: T.Buffer(shape_v, dtype), # type: ignore
            {{final_rowscales_output | indent(12)}}
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dimv], dtype)
                # V_local = T.alloc_fragment([block_N, dimv], dtype)
                scores = T.alloc_fragment([block_M, block_N], accum_dtype)
                scores_shared = T.alloc_shared([block_M, block_N], accum_dtype)
                scores_1 = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_s_cast_1 = T.alloc_fragment([block_M, block_N], dtype)
                # acc_o = T.alloc_fragment([block_M, dimv], accum_dtype)

                {{custom_fwd_inputs_init | indent(16)}}
                
                # block_mask = T.alloc_local([seq_len_kv//block_N], block_mask_dtype)
                has_valid_block = T.alloc_var("bool")
                q_idxl = T.alloc_fragment([1], "int32")
                kv_idxl = T.alloc_fragment([1], "int32")

                T.annotate_layout({
                    Q_shared: tl.layout.make_swizzled_layout(Q_shared),
                    scores_shared: tl.layout.make_swizzled_layout(scores_shared),
                    {{swizzle_shared | indent(20)}}
                })
                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                {{custom_fwd_inputs_load_prolog | indent(16)}}
                T.fill(acc_o, 0)
                T.fill({{o_scale_varname}}, 1.0)

                {{online_rowscales_initvalue | indent(16)}}

                # for vj in T.serial(seq_len_kv // block_N):
                #     block_mask[vj] = BlockSparseMask[bz, by, bx//block_M_ratio, vj//block_N_ratio]

                # TODO: mask
                loop_range = (
                    T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
                )
                has_valid_block = False

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if BlockSparseMask[bz, by, bx//block_M_ratio, k//block_N_ratio] != 0:
                        has_valid_block = True
                        T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)

                        # TODO: copy custom_fwd_input_tensor in score_mod&online_func
                        {{custom_fwd_inputs_load_shared | indent(24)}}

                        # TODO: naive solution: if reduce_max, -T.inf; if reduce_sum, 0
                        if (is_casual or {{is_mask_mod_code}}) and {{is_inf_mask}}:
                            for i, j in T.Parallel(block_M, block_N):
                                q_idxl[0] = bx * block_M + i
                                kv_idxl[0] = k * block_N + j
                                {{q_idx}} = q_idxl[0]
                                {{kv_idx}} = kv_idxl[0]
                                # {{q_idx}} = bx * block_M + i
                                # {{kv_idx}} = k * block_N + j
                                {{batch_idx}} = bz
                                {{head_idx}} = by
                                {{mask_mod_code | indent(32)}}
                                scores[i, j] = T.if_then_else(
                                    {{mask_output}}, 0, -T.infinity(scores.dtype)
                                )
                        else:
                            T.clear(scores)
                        
                        T.gemm(Q_shared, K_shared, scores, transpose_B=True, policy= (T.GemmWarpPolicy.FullRow if (not shared_fuse) else T.GemmWarpPolicy.FullCol))
                        T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                            
                        {{custom_fwd_inputs_load_s2r | indent(24)}}
                        # call score_mod
                        {{call_score_mod | indent(24)}}
                            
                        # call online_func
                        if shared_fuse:
                            T.copy(scores, scores_shared)
                            T.copy(scores_shared, scores_1)
                            {{call_online_func | indent(28)}}
                            T.copy(scores_1, acc_s_cast_1)

                        else:
                            {{call_online_func | indent(28)}}
                            T.copy(scores, acc_s_cast)

                        for i, j in T.Parallel(block_M, dimv):
                            acc_o[i, j] *= {{o_scale_varname}}[i]
                        
                        # update online_rowscales
                        {{online_rowscales_update | indent(24)}}

                        if shared_fuse:
                            T.gemm(acc_s_cast_1, V_shared, acc_o, policy=(T.GemmWarpPolicy.FullCol))
                        else:
                            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                # online_fwd_epilogue
                if has_valid_block:
                    {{online_func_epilogue | indent(20)}}

                T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

                # save final_rowscale
                {{final_rowscales_save | indent(16)}}
            
        return main
    
    if tune:
        @autotune(
            configs=get_configs(dim, dimv, device=attn_device),
            warmup=10,
            rep=10,
        )
        @jit(out_idx={{output_idx_list}}, supply_type=tilelang.TensorSupplyType.Auto, ref_prog=None)
        def kernel(block_M=None, block_N=None, num_stages=None, thread_num=None, shared_fuse=None):
            return kernel_func(block_M, block_N, num_stages, thread_num, shared_fuse)

        return kernel()

    else:
        def kernel(block_M, block_N, num_stages, thread_num, shared_fuse):
            return kernel_func(block_M, block_N, num_stages, thread_num, shared_fuse)
        
        return kernel

# TL_KERNEL_BWD_DOO = """
def flashattn_bwd_preprocess(batch, heads, seq_len, dim, dimv):
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
        O: T.Buffer(shape_v, dtype), # type: ignore
        dO: T.Buffer(shape_v, dtype), # type: ignore
        Delta: T.Buffer([batch, heads, seq_len], accum_dtype), # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dimv, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep

# TL_KERNEL_BWD = """
def flashattn_bwd(batch, heads, seq_len, dim, dimv, is_casual, 
                downsample_len_q, downsample_len_kv,
                block_M, block_N, thread_num = 128*2):
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    # TODO: seqlenkv
    seq_len_kv = seq_len
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    
    assert(downsample_len_kv <= seq_len_kv // block_N)
    assert(downsample_len_q <= seq_len // block_M)
    block_M_ratio = seq_len_kv // block_M // downsample_len_q
    block_N_ratio = seq_len // block_N // downsample_len_kv
    block_mask_shape = [batch, heads, downsample_len_q, downsample_len_kv]
    block_mask_dtype = "int8"

# TL_MAIN_BWD = """
    @T.macro
    def score_mod(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{score_mod_fwd_inputs | indent(8)}}
        ):
        {{score_mod_fwd_body | indent(8)}}
        pass
    
    @T.macro
    def score_mod_backward(
        # scores: T.Buffer([block_M, block_N], accum_dtype),
        {{score_mod_bwd_inputs | indent(8)}}
    ):
        {{score_mod_backward | indent(8)}}
        pass

    @T.prim_func
    def flash_bwd(
        Q: T.Buffer(shape, dtype), # type: ignore
        K: T.Buffer(shape, dtype), # type: ignore
        V: T.Buffer(shape_v, dtype), # type: ignore
        dO: T.Buffer(shape_v, dtype), # type: ignore

        # custom_fwd_inputs score_mod
        {{custom_fwd_inputs | indent(8)}}

        # final_rowscales
        {{final_rowscales_output | indent(8)}}

        # custom_bwd_inputs
        {{custom_bwd_inputs | indent(8)}}

        BlockSparseMask: T.Buffer(block_mask_shape, block_mask_dtype), # type: ignore
        dQ: T.Buffer(shape, accum_dtype), # type: ignore
        dK: T.Buffer(shape, dtype), # type: ignore
        dV: T.Buffer(shape_v, dtype), # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=thread_num) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            # should not store K to local if dim is large
            # K_local = T.alloc_fragment([block_M, dim], dtype)
            # H100 wgmma
            # K_local_T = T.alloc_fragment([block_M, dim], dtype)
            # V_local = T.alloc_fragment([block_M, dimv], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_M, dimv], dtype)
            # qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            # dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)

            # final_rowscales_declare
            {{final_rowscales_shared_init | indent(12)}}

            # custom_bwd_declare
            {{custom_bwd_inputs_init | indent(12)}}

            # score_mod_declare
            {{score_mod_bwd_inputs_declare | indent(12)}}
            # score_mod_declare_shared
            {{score_mod_bwd_inputs_declare_shared | indent(12)}}
            
            do = T.alloc_shared([block_N, dimv], dtype)
            dv = T.alloc_fragment([block_M, dimv], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)
            dv_shared = T.alloc_shared([block_N, dimv], dtype)
            dk_shared = T.alloc_shared([block_N, dim], dtype)
            T.annotate_layout(
                {
                    dQ: make_dq_layout(dQ),
                    K_shared: tl.layout.make_swizzled_layout(K_shared),
                    dv_shared: tl.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tl.layout.make_swizzled_layout(dk_shared),
                }
            )
            T.copy(K[bz, by * block_M : (by + 1) * block_M, bx, :], K_shared)
            T.copy(V[bz, by * block_M : (by + 1) * block_M, bx, :], V_shared)
            # T.copy(K_shared, K_local)
            # T.copy(K_shared, K_local_T)
            # custom_fwd_inputs_load_prolog
            {{custom_fwd_inputs_load_prolog | indent(12)}}
            T.clear(dv)
            T.clear(dk)

            # TODO: is causal
            loop_st = T.floordiv(by * block_M, block_N) if is_casual else 0
            loop_ed = T.ceildiv(seq_len, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                if BlockSparseMask[bz, bx, k//block_N_ratio, by//block_M_ratio] !=0:
                    T.copy(Q[bz, k * block_N : (k + 1) * block_N, bx, :], q)
                    {{custom_fwd_inputs_load_shared_bwd | indent(20)}}
                    T.clear(qkT)
                    T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    # score_mod
                    score_mod({{score_mod_inputs_bwd_list}}) # qkT,

                    # final_rowscales_load
                    {{final_rowscales_load | indent(20)}}

                    # online_func_fwd
                    {{ online_func_fwd | indent(20) }}
                    
                    # TODO: is causal
                    if is_casual or {{is_mask_mod_code}}:
                        for i, j in T.Parallel(block_M, block_N):
                            {{q_idx}} = k * block_N + j
                            {{kv_idx}} =  by * block_M + i
                            {{batch_idx}} = bz
                            {{head_idx}} = bx
                            {{mask_mod_code | indent(28)}}
                            {{score_mod_output_var}}[i, j] = T.if_then_else(
                                {{mask_output}}, {{score_mod_output_var}}[i, j], 0
                            )
                    
                    T.copy(dO[bz, k * block_N : (k + 1) * block_N, bx, :], do)
                    T.clear(dsT)
                    T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    T.copy({{score_mod_output_var}}, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                    # custom_bwd_inputs_load
                    {{custom_bwd_inputs_load | indent(20)}}

                    # T.clear(dsT)
                    # T.gemm(V_local, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    if is_casual or {{is_mask_mod_code}}:
                        for i, j in T.Parallel(block_M, block_N):
                            {{q_idx}} = k * block_N + j
                            {{kv_idx}} =  by * block_M + i
                            {{batch_idx}} = bz
                            {{head_idx}} = bx
                            {{mask_mod_code | indent(28)}}
                            dsT[i, j] = T.if_then_else(
                                {{mask_output}}, dsT[i, j], 0
                            )

                    # custom_bwd
                    {{custom_bwd_body | indent(20)}}
                    
                    # score_mod_backward
                    score_mod_backward({{score_mod_bwd_inputs_list}}) #  qkT, 
                    
                                    
                    T.copy(dsT, dsT_cast)
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    # T.gemm(dsT_shared, K_local_T, dq, transpose_A=True)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)
                    for i, j in T.Parallel(block_N, dim):
                        if k * block_N + i < seq_len:
                            T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])
            T.copy(dv, dV[bz, by * block_M : (by + 1) * block_M, bx, :])
            T.copy(dk, dK[bz, by * block_M : (by + 1) * block_M, bx, :])

    return flash_bwd              

# TL_KERNEL_BWD_POSTPROCESS = """
def flashattn_bwd_postprocess(batch, heads, seq_len, dim, dimv):
    dtype = "{{tl_dtype}}" # "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    shape_v = [batch, seq_len, heads, dimv]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
        dQ: T.Buffer(shape, accum_dtype), # type: ignore
        dQ_out: T.Buffer(shape, dtype), # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_post

TUNE = {{TUNE}}
TUNE_FILE = "{{TUNE_FILE}}"
TUNE_BWD = {{TUNE_BWD}}
TUNE_FILE_BWD = "{{TUNE_FILE_BWD}}"

def get_problem_keys():
    return {
        "batch": 4 if isinstance({{BATCH}}, T.Var) else {{BATCH}},
        "heads": 32 if isinstance({{HEADS}}, T.Var) else {{HEADS}},
        "seq_len": 2048 if isinstance({{SEQ_LEN}}, T.Var) else {{SEQ_LEN}},
        "dim": {{DIM}},
        "dimv": {{DIMV}},
        "downsample_len_q": {{SEQ_LEN}} // {{infer_mask_block_M}},
        "downsample_len_kv": {{SEQ_LEN}} // {{infer_mask_block_N}},
    }
    
def tune(tune_file, kernel_profiler, problem_keys)->Tuple:
    tuned_config = None
    pk = problem_keys
    
    folder_path = os.path.dirname(tune_file)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    if not os.path.exists(tune_file):
        with open(tune_file, "w") as f:
            json.dump([], f, indent=4)
        configs = []
    else:
        with open(tune_file, "r") as f:
            configs = json.load(f)
        # find the config with the same problem size
        for config in configs:
            if all(config[k] == v for k, v in pk.items()):
                tuned_config = config['tuned_config']
                break
    if tuned_config is None:
        result = kernel_profiler(
            **problem_keys
        )
        tuned_config = result.config
        with open(tune_file, "w") as f:
            configs.append({
                **pk,
                'tuned_config': tuned_config
            })
            json.dump(configs, f, indent=4)
    return tuned_config

    
# forward
tuned_config = None
if TUNE:
    pk = get_problem_keys()
    _tuned_config = tune(TUNE_FILE, partial(kernel, tune=True), pk)
    tuned_config = {
        'block_M': _tuned_config[0],
        'block_N': _tuned_config[1],
        'num_stages': _tuned_config[2],
        'thread_num': _tuned_config[3],
        'shared_fuse': _tuned_config[4]
    }
else:
    tuned_config = {
        'block_M': {{block_M}},
        'block_N': {{block_N}},
        'num_stages': {{stages}},
        'thread_num': {{thread_num}},
        'shared_fuse': {{shared_fuse}}
    }    
block_mask_M = {{infer_mask_block_M}}# 128
block_mask_N = {{infer_mask_block_N}}# 128
downsample_len_q = {{SEQ_LEN}} // block_mask_M
downsample_len_kv = {{SEQ_LEN}} // block_mask_N

program = kernel({{BATCH}}, {{HEADS}}, {{SEQ_LEN}}, {{DIM}}, {{DIMV}}, downsample_len_q, downsample_len_kv)(**tuned_config)
mod = tl.compile(program, out_idx={{output_idx_list}})# , execution_backend="dlpack")# , pass_configs={"tl.disable_tma_lower": True})
        
mod_prep = tl.compile(
    flashattn_bwd_preprocess({{BATCH}}, {{HEADS}}, {{SEQ_LEN}}, {{DIM}}, {{DIMV}}), 
    out_idx=[2],
)
mod_post = tl.compile(
    flashattn_bwd_postprocess({{BATCH}}, {{HEADS}}, {{SEQ_LEN}}, {{DIM}}, {{DIMV}}), 
    out_idx=[1],
)
mod_bwd = tl.compile(
    flashattn_bwd(
        {{BATCH}}, {{HEADS}}, {{SEQ_LEN}}, {{DIM}}, {{DIMV}}, {{is_casual}}, 
        downsample_len_q, downsample_len_kv, {{block_M_bwd}}, {{block_N_bwd}}, {{thread_num_bwd}}
    ),
    out_idx={{bwd_output_idx_list}},
)

# TL_INFERFACE = """
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, *custom_fwd_inputs):
        custom_fwd_inputs, block_sparse_mask = custom_fwd_inputs[:-1], custom_fwd_inputs[-1]
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEADV = v.shape[-1]

        output_idx_list = {{output_idx_list}}
        global mod
        if len(output_idx_list) == 1:
            o = mod(q, k, v, *custom_fwd_inputs, block_sparse_mask)
            final_scale = []
        else:
            o, *final_scale = mod(q, k, v, *custom_fwd_inputs, block_sparse_mask)
        ctx.save_for_backward(q, k, v, o, block_sparse_mask, *custom_fwd_inputs, *final_scale)
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, block_sparse_mask, *tmp = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD = q.shape
        D_HEAD_V = v.shape[-1]
        # custom_fwd_inputs = tmp[:-{{final_rowscales_length}}]
        # final_rowscales = tmp[-{{final_rowscales_length}}:]
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        
        global mod_prep, mod_post, mod_bwd
        if {{isused_doosum}}:
            delta = mod_prep(o, do)
        # TODO: causal
        
        dq = torch.zeros_like(q, dtype=torch.float32)
        if {{isused_doosum}}:
            dk, dv = mod_bwd(q, k, v, do, *tmp, delta, block_sparse_mask, dq)
        else:
            dk, dv = mod_bwd(q, k, v, do, *tmp, block_sparse_mask, dq)
        dq = mod_post(dq)
        none_list = [None] * len(tmp)
        return dq, dk, dv, *none_list, None

attention = _attention.apply


