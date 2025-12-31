import torch
import tilelang as tl
import tilelang
import tilelang.language as T

# from tilelang.autotuner import *

import triton.language as triton_lang
import triton

import itertools
import os
import json
from typing import Tuple
from functools import partial
import multiprocessing
import traceback

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

# ---------------------TRITON_KERNEL_CUMSUM
@triton.jit
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    T: triton_lang.constexpr,
    BT: triton_lang.constexpr,
):
    i_t, i_bh = triton_lang.program_id(0), triton_lang.program_id(1)
    p_s = triton_lang.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_o = triton_lang.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    # [BT, BS]
    b_s = triton_lang.load(p_s, boundary_check=(0,)).to(triton_lang.float32)
    b_o = triton_lang.cumsum(b_s, axis=0)
    triton_lang.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))

def chunk_local_cumsum_scalar(g, BT):
    B, H, T = g.shape
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        g_org, g,
        T=T, BT=BT
    )
    return g

# ---------------------TRITON_KERNEL_REV_CUMSUM
@triton.jit
def compute_final_dg(
    dg,
    o,
    T: triton_lang.constexpr,
    BT: triton_lang.constexpr,
):
    i_t, i_bh = triton_lang.program_id(0), triton_lang.program_id(1)

    p_o = triton_lang.make_block_ptr(dg + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_o = triton_lang.load(p_o, boundary_check=(0,))
    b_o = b_o - triton_lang.cumsum(b_o, axis=0) + triton_lang.sum(b_o, axis=0)
    p_o = triton_lang.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    triton_lang.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))



def generate_config_h(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    BK_hs = [32,64,128] # ,192,256]
    BV_hs = [32,64,128] # ,192,256]
    num_stages_hs = [1,2] # ,3,4]
    num_threads_hs = [128,256]
    
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    # BTs = [bt for bt in BTs if N_CTX % bt == 0]
    BK_hs = [bk for bk in BK_hs if D_HEAD % bk == 0]
    BV_hs = [bv for bv in BV_hs if D_HEADV % bv == 0]
       
    config_h = []
    for BK_h, BV_h, num_stages_h, num_threads_h in itertools.product(BK_hs, BV_hs, num_stages_hs, num_threads_hs):
        sharedmem_chunk_h = 2 * (
            BK_h * BV_h + (BT * BK_h + BT * BV_h + BT ) 
            * num_stages_h)
        reg_chunk_h = 4 * (
            BK_h * BV_h
        )
        conditions_h = [
            num_stages_h > N_CTX // BT,
            BK_h % (MMA_ATOM_M) != 0,
            sharedmem_chunk_h > smem_cap,
            reg_chunk_h > reg_cap and reg_chunk_h > reg_cap_per_thread * num_threads_h,
        ]
        if any(conditions_h):
            continue
        config_h.append( {
        # "BT": BT,
        "BK": BK_h,
        "BV": BV_h,
        "num_stages": num_stages_h,
        "num_threads": num_threads_h,
        })
    return config_h

# --------------- TL_KERNEL_H
def chunk_fwd_h(
        batch, headq, headk, head, seqlen, dim, dimv, BT, tune=False):
    # BT = 64
    # BK = 64
    # BV = 64
    dtype = "bfloat16" # mi250: float16
    accum_dtype = "float"
    LOG2E = 1.44269504
    
    seq_len = seqlen
    heads = head
    dimqk = dim

    assert(head % headk == 0)
    head_headk_ratio = head // headk
    
    def kernel_func(BK, BV, num_stages, num_threads):
        NT = seqlen // BT
        NK = dim // BK
        NV = dimv // BV
        num_stages = num_stages

        @T.prim_func
        def main(
            k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
            v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
            g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
            {{chunk_h_custom_inputs_list | indent(12)}}
            h: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
        ):
            with T.Kernel(NK, NV, batch * head, threads=num_threads) as (bx, by, bz):

                b_h = T.alloc_fragment((BK, BV), accum_dtype)
                # b_h_cast = T.alloc_fragment((BK, BV), dtype)
                b_h_shared = T.alloc_shared((BK, BV), dtype)
                b_k = T.alloc_fragment((BT, BK), dtype)
                b_k_shared = T.alloc_shared((BT, BK), dtype)
                b_kt = T.alloc_fragment((BK, BT), dtype)
                b_v_shared = T.alloc_shared((BT, BV), dtype)
                b_v = T.alloc_fragment((BT, BV), dtype)
                b_g = T.alloc_fragment((BT), accum_dtype)
                b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
                b_glast = T.alloc_fragment((1), accum_dtype)
                {{h_alloc_buffer_list | indent(16)}}

                bhead = bz % head
                bb = bz // head

                bheadk = bhead // head_headk_ratio
                
                T.annotate_layout({
                    # b_v_shared: tl.layout.make_swizzled_layout(b_v_shared),
                    b_k_shared: tl.layout.make_swizzled_layout(b_k_shared),
                    b_h_shared: tl.layout.make_swizzled_layout(b_h_shared),
                }
                )
                T.clear(b_h)

                for i_t in T.Pipelined(NT, num_stages=num_stages):
                    # T.copy(b_h, b_h_shared)
                    # T.copy(h[bb,bhead,(i_t*dim+bx*BK):(i_t*dim+(bx+1)*BK), by*BV:(by+1)*BV], b_h)
                    # T.copy(k[bb,bhead,i_t*BT:(i_t+1)*BT,bx*BK:(bx+1)*BK], b_k)
                    T.copy(k[bb,bheadk,i_t*BT:(i_t+1)*BT,bx*BK:(bx+1)*BK], b_k_shared)
                    T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v_shared)
                    T.copy(g[bb,bhead,i_t*BT:(i_t+1)*BT], b_g_shared)
                    # T.copy(v[bb,bhead,i_t*BT:(i_t+1)*BT,by*BV:(by+1)*BV], b_v)
                    # T.copy(b_v, b_v_shared)
                    # T.copy(g[bb,bhead,i_t*BT:(i_t+1)*BT], b_g)
                    # T.copy(g[bb,bhead,(i_t+1)*BT-1:(i_t+1)*BT], b_glast)
                    b_glast[0] = g[bb,bhead,(i_t+1)*BT-1]

                    T.copy(b_h, b_h_shared) # implicit cast
                    T.copy(b_h_shared, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
                    # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])

                    # scalar_decay
                    # for i0 in T.Parallel(BT):
                    #     b_g[i0] = T.exp(b_g[i0])

                    for i0, i1 in T.Parallel(BK, BV):
                        b_h[i0, i1] = b_h[i0, i1] * T.exp2(b_glast[0] * LOG2E)
                    
                    # for i0,i1 in T.Parallel(BT, BV):
                    #     b_v_shared[i0,i1] *= T.exp2((b_glast[0] - b_g[i0]) * 1.44269504)

                    T.copy(b_k_shared, b_k)
                    
                    {{k_mod_expr_fused_h | indent(20)}}
                    
                    T.copy(b_g_shared, b_g)
                    for i0, i1 in T.Parallel(BK, BT):
                        b_kt[i0, i1] = b_k[i1, i0]*T.exp2((b_glast[0] - b_g[i1]) * LOG2E)

                    T.gemm(b_kt, b_v_shared, b_h, transpose_B=False)
                    # T.copy(b_h, b_h_shared)
                    
                    # TODO
                    # T.copy(b_h, h[bb,bhead,i_t*dim+bx*BK:(i_t)*dim+(bx+1)*BK,by*BV:(by+1)*BV])
        
        return main
    
    if tune:
        configs = generate_config_h(batch,headq,headk,head,seqlen,dim,dimv,BT, device=attn_device)
        if len(configs) == 0:
            return None
        @tilelang.autotune(
            configs=configs,
            warmup=10,
            rep=10,
        )
        @tilelang.jit(out_idx={{output_idx_list_h}})
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel()
    else:
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel
        

def generate_config_o(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    BK_os = [32,64,128] # ,256]
    BV_os = [32,64,128] # ,256]
    num_stages_os = [1,2] # ,3,4]
    num_threads_os = [128,256]
    
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    # BTs = [bt for bt in BTs if N_CTX % bt == 0]
    BK_os = [bk for bk in BK_os if D_HEAD % bk == 0]
    BV_os = [bv for bv in BV_os if D_HEADV % bv == 0]
    
    config_o = []
    for BK_o, BV_o, num_stages_o, num_threads_o in itertools.product(BK_os, BV_os, num_stages_os, num_threads_os):
        sharedmem_chunk_o = 2 * (
            BT * BK_o * num_stages_o + BT * BK_o * num_stages_o + BK_o * BV_o* num_stages_o +
            BT * BV_o 
        )
        reg_chunk_o = 4 * (
            BT * BT + BT * BV_o
        )
        conditions_o = [
            num_stages_o > D_HEAD // BK_o,
            BT % (MMA_ATOM_M*(num_threads_o//MMA_ATOM_TRHEADS)) != 0,
            sharedmem_chunk_o > smem_cap,
            reg_chunk_o > reg_cap and reg_chunk_o > reg_cap_per_thread * num_threads_o,
        ]
        if any(conditions_o):
            continue
        config_o.append( {
        # "BT": BT,
        "BK": BK_o,
        "BV": BV_o,
        "num_stages": num_stages_o,
        "num_threads": num_threads_o,
        })
    return config_o
        

# --------------- TL_KERNEL_O 
def chunk_o(
        batch, headq,headk, head, seqlen, dim, dimv, BT, tune=False):

    # BT = 64
    # BK = 64
    # BV = 64
    NT = seqlen // BT
    dtype = "bfloat16" # mi250: float16
    accum_dtype = "float"
    LOG2E = 1.44269504

    scale = 1.0
    seq_len = seqlen
    heads = head
    dimqk = dim
    
    BT2 = BT

    assert(head % headk == 0)
    head_headk_ratio = head // headk
    assert(head % headq == 0)
    head_headq_ratio = head // headq
    
    BT2 = BT

    def kernel_func(BK, BV, num_stages, num_threads):
        NK = dim // BK
        NV = dimv // BV
        
        @T.prim_func
        def main(
            h: T.Buffer((batch,head,NT*dim,dimv), dtype), # type: ignore
            q: T.Buffer((batch,headq,seqlen,dim), dtype), # type: ignore
            k: T.Buffer((batch,headk,seqlen,dim), dtype), # type: ignore
            v: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
            g: T.Buffer((batch,head,seqlen), accum_dtype), # type: ignore
            {{chunk_o_custom_inputs_list | indent(12)}}
            o: T.Buffer((batch,head,seqlen,dimv), dtype), # type: ignore
            # custom fwd inputs
        ):
            with T.Kernel(NV, NT, batch * head, threads=num_threads) as (bx, by, bz):
                bo = T.alloc_fragment((BT, BV), dtype=accum_dtype)
                bo_shared = T.alloc_shared((BT, BV), dtype=dtype)
                bs = T.alloc_fragment((BT, BT2), dtype=accum_dtype)
                bs_cast = T.alloc_fragment((BT, BT2), dtype=dtype)
                bq = T.alloc_fragment((BT, BK), dtype=dtype)
                bq_shared = T.alloc_shared((BT,BK), dtype=dtype)
                bk_shared = T.alloc_shared((BT,BK), dtype=dtype)
                bv_shared = T.alloc_shared((BT,BV), dtype=dtype)
                b_state_shared = T.alloc_shared((BK, BV), dtype=dtype)
                bg_shared = T.alloc_shared((BT,), dtype=accum_dtype, scope = "shared")
                bg = T.alloc_fragment((BT,), dtype=accum_dtype)
                bg1 = T.alloc_fragment((BT,), dtype=accum_dtype)
                {{o_alloc_buffer_list | indent(16)}}
                # custom fwd inputs init

                bb = bz // head
                bh = bz % head

                bhk = bh // head_headk_ratio
                bhq = bh // head_headq_ratio

                T.annotate_layout({
                    bq_shared: tl.layout.make_swizzled_layout(bq_shared),
                    bo_shared: tl.layout.make_swizzled_layout(bo_shared),
                })
                T.clear(bo)
                T.clear(bs)
                for ik in T.Pipelined(NK, num_stages=num_stages):
                    # pipeline here
                    T.copy(q[bb, bhq, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq_shared)
                    # T.copy(q[bb, bh, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bq)
                    T.copy(k[bb, bhk, by*BT:(by+1)*BT, ik*BK:(ik+1)*BK], bk_shared)

                    T.copy(h[bb, bh, by*dim+ik*BK:by*dim+(ik+1)*BK, bx*BV:(bx+1)*BV], b_state_shared)
                    
                    T.copy(bq_shared, bq)
                    # q_mod here (fused)
                    {{q_mod_expr | indent(20)}}

                    T.gemm(bq, bk_shared, bs, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(bq, b_state_shared, bo, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
                
                # T.copy(g[bb, bh, by*BT:(by+1)*BT], bg)
                # T.copy(g[bb, bh, by*BT:(by+1)*BT], bg1)
                T.copy(g[bb, bh, by*BT:(by+1)*BT], bg_shared)
                T.copy(bg_shared, bg)
                T.copy(bg_shared, bg1)
                for i0,i1 in T.Parallel(BT,BV):
                    bo[i0,i1] *= T.exp2(bg[i0] * LOG2E)
                
                for i0,i1 in T.Parallel(BT,BT2):
                    bs[i0,i1] *= T.exp2((bg[i0]-bg1[i1]) * LOG2E)
                # fix nan bug
                for i0,i1 in T.Parallel(BT,BT2):
                    bs[i0,i1] = T.if_then_else(
                        i0 >= i1, bs[i0,i1], 0.0
                    )

                # v_mod here (fused)
                {{v_mod_expr_fused_o | indent(16)}}
                
                T.copy(v[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV], bv_shared)
                T.copy(bs, bs_cast)
                T.gemm(bs_cast, bv_shared, bo, policy=T.GemmWarpPolicy.FullRow)
                # T.copy(bo, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV]) # slow for stride between thread
                T.copy(bo, bo_shared) # implicit type convert
                T.copy(bo_shared, o[bb, bh, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
        
        return main
    
    if tune:
        configs = generate_config_o(batch,headq,headk,head,seqlen,dim,dimv,BT, device=attn_device)
        if len(configs) == 0:
            return None
        @tilelang.autotune(
            configs=configs,
            warmup=10,
            rep=10,
        )
        @tilelang.jit(out_idx={{output_idx_list_o}})
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel()
    else:
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel
        

def generate_config_dh(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    BK_dhs = [32,64] if device.platform=="ROCM" else [32, 64,128] # ,192,256]
    BV_dhs = [32,64] if device.platform=="ROCM" else [32,64,128] # ,192,256]
    num_stages_dhs = [1,2] if device.platform=="CUDA" else [1,] # ,3,4]
    num_threads_dhs = [128,256]
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    BK_dhs = [bk for bk in BK_dhs if D_HEAD % bk == 0]
    BV_dhs = [bv for bv in BV_dhs if D_HEADV % bv == 0]
    config_dh = []
    for BK_dh, BV_dh, num_stages_dh, num_threads_dh in itertools.product(BK_dhs, BV_dhs, num_stages_dhs, num_threads_dhs):
        # bug mi250
        if BT >= 128 and BK_dh > 64 and device.platform == "ROCM":
            continue
        conditions_dh = [
            num_stages_dh > N_CTX // BT,
            BK_dh % (MMA_ATOM_M) != 0,
            # sharedmem_chunk_dh > smem_cap,
            # reg_chunk_dh > reg_cap and reg_chunk_dh > reg_cap_per_thread * num_threads_dh,
        ]
        if any(conditions_dh):
            continue
        config_dh.append( {
            "BK": BK_dh,
            "BV": BV_dh,
            "num_stages": num_stages_dh,
            "num_threads": num_threads_dh,
        })
    return config_dh
    
# --------------- TL_KERNEL_BWD_dh
def chunk_bwd_kernel_dh(
    batch, headq, headk, head, seqlen, dim, dimv, BT, tune=False): 
    
    def kernel_func(
        BK, BV, num_stages, num_threads):
        NT = seqlen // BT
        NK = dim // BK
        NV = dimv // BV
        dtype = "bfloat16" # mi250 "float16"
        accum_dtype = "float"
        num_stages = num_stages
        thread_num = num_threads

        scale = 1.0
        assert(head % headk == 0)
        head_headk_ratio = head // headk
        assert(head % headq == 0)
        head_headq_ratio = head // headq

        @T.prim_func
        def main(
            q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
            k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
            v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
            g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
            # must not the same with preserved "do"
            d_o: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
            
            dh: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
        ):
            with T.Kernel(NK, NV, batch*head, threads=thread_num) as (bx, by, bz):
                b_dh = T.alloc_fragment((BK, BV), accum_dtype)
                dh_shared = T.alloc_shared((BK, BV), dtype)
                q_shared = T.alloc_shared((BT, BK), dtype)
                q_local = T.alloc_fragment((BT, BK), dtype)
                q_local_T = T.alloc_fragment((BK, BT), dtype)
                do_shared = T.alloc_shared((BT, BV), dtype)

                bg_local = T.alloc_fragment((BT), accum_dtype)
                bg_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
                bg_last = T.alloc_fragment((1), accum_dtype)

                bhead = bz % head
                bb = bz // head
                bheadk = bhead // head_headk_ratio
                bheadq = bhead // head_headq_ratio

                T.clear(b_dh)
                loop_st = 0
                loop_ed = NT
                for i_t0 in T.Pipelined(NT, num_stages=num_stages):
                    i_t = loop_ed - 1 - i_t0

                    # T.copy(b_dh, dh[bb, bhead, (i_t*dim + bx*BK):(i_t*dim + (bx+1)*BK), by*BV:(by+1)*BV]) # implicit cast
                    T.copy(b_dh, dh_shared)
                    T.copy(dh_shared,  dh[bb, bhead, (i_t*dim + bx*BK):(i_t*dim + (bx+1)*BK), by*BV:(by+1)*BV])

                    T.copy(q[bb, bheadq, i_t*BT:(i_t+1)*BT, bx*BK:(bx+1)*BK], q_shared)
                    T.copy(d_o[bb, bhead, i_t*BT:(i_t+1)*BT, by*BV:(by+1)*BV], do_shared)
                    
                    T.copy(q_shared, q_local)

                    T.copy(g[bb, bhead, i_t*BT:(i_t+1)*BT], bg_shared)

                    # q_mod 
                    for i,j in T.Parallel(BT, BK):
                        q_local[i,j] *= scale

                    T.copy(bg_shared, bg_local)
                    bg_last[0] = bg_shared[BT-1]
                    for i,j in T.Parallel(BK, BT):
                        q_local_T[i,j] = q_local[j,i] * T.exp(bg_local[j])

                    for i,j in T.Parallel(BK, BV):
                        b_dh[i,j] *= T.exp(bg_last[0])
                    
                    T.gemm(q_local_T, do_shared, b_dh, transpose_B=False)
            
        return main
    
    if tune:
        configs = generate_config_dh(batch,headq,headk,head,seqlen,dim,dimv,BT, device=attn_device)
        if len(configs) == 0:
            return None
        @tilelang.autotune(
            configs=configs,
            warmup=10,
            rep=10,
        )
        @tilelang.jit(out_idx=[5,])
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel()
    else:
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel

def generate_config_dqkg(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    BK_dqkg = [32,64,128] if BT<128 or device.platform=="CUDA" else [32,64]
    BV_dqkg = [32,64,128] if BT<128 or device.platform=="CUDA" else [32,64]
    num_stages_dqkg = [1,2] if device.platform=="CUDA" else [1]
    num_threads_dqkg = [128,256]
    
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    BK_dqkg = [bk for bk in BK_dqkg if D_HEAD % bk == 0]
    BV_dqkg = [bv for bv in BV_dqkg if D_HEADV % bv == 0]
    
    def is_power_of_two(n):
        return (n & (n - 1)) == 0 and n > 0
    
    config_dqkg = []
    for BK_dqk, BV_dqk, num_stages_dqk, num_threads_dqk in itertools.product(BK_dqkg, BV_dqkg, num_stages_dqkg, num_threads_dqkg):
        # mi250 bug
        if device.platform == "ROCM" and BT >= 128 and (BK_dqk > 32 or BV_dqk > 32):
            continue
        conditions_dqk = [
            num_stages_dqk > D_HEADV // BV_dqk,
            BT % (MMA_ATOM_M*(num_threads_dqk//MMA_ATOM_TRHEADS)) != 0,
            # Tilelang 0.1.5 T.reduce_sum limit
            not is_power_of_two(BT // (num_threads_dqk // device.warp_size)),
            # sharedmem_chunk_dqk > smem_cap,
            # reg_chunk_dqk > reg_cap and reg_chunk_dqk > reg_cap_per_thread * num_threads_dqk,
        ]
        if any(conditions_dqk):
            continue
        config_dqkg.append( {
        # "BT": BT,
        "BK": BK_dqk,
        "BV": BV_dqk,
        "num_stages": num_stages_dqk,
        "num_threads": num_threads_dqk,
        })
    return config_dqkg

# --------------- TL_KERNEL_BWD_dqkg
def chunk_bwd_dqkg(
    batch, headq, headk, head, seqlen, dim, dimv, BT, tune=False,):
    
    def kernel_func(
        BK, BV, num_stages, num_threads
    ):
        NT = seqlen // BT
        NK = dim // BK
        NV = dimv // BV
        dtype = "bfloat16" # mi250 "float16"
        accum_dtype = "float"
        num_stages = num_stages
        thread_num = num_threads

        scale = 1.0
        assert(head % headk == 0)
        head_headk_ratio = head // headk
        assert(head % headq == 0)
        head_headq_ratio = head // headq

        @T.prim_func
        def main(
            q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
            k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
            v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
            h: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
            g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
            d_o: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
            dh:  T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore

            dq: T.Buffer((batch, head, seqlen, dim), dtype), # type: ignore
            dk: T.Buffer((batch, head, seqlen, dim), dtype), # type: ignore
            dg: T.Buffer((NK, batch, head, seqlen), accum_dtype), # type: ignore
        ):
            with T.Kernel(NK, NT, batch*head, threads=thread_num) as (bx, by, bz):
                b_g = T.alloc_fragment((BT), accum_dtype)
                b_g1 = T.alloc_fragment((BT), accum_dtype)
                b_g_last = T.alloc_fragment((1), accum_dtype)
                b_g_T = T.alloc_fragment((BT), accum_dtype)
                b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
                b_v_shared = T.alloc_shared((BT, BV), dtype)
                b_do_shared = T.alloc_shared((BT, BV), dtype)
                b_dh_shared = T.alloc_shared((BK, BV),dtype)
                b_dh_local = T.alloc_fragment((BK, BV),dtype)
                b_h_shared = T.alloc_shared((BK, BV),dtype)
                b_h_local = T.alloc_fragment((BK, BV),dtype)
                b_hdh = T.alloc_fragment((1,BK*BV),accum_dtype)
                b_hdh2 = T.alloc_fragment((BK, BV),accum_dtype)
                k_shared = T.alloc_shared((BT,BK), dtype)
                q_shared = T.alloc_shared((BT,BK),dtype)
                q_local = T.alloc_fragment((BT,BK),dtype)
                k_local = T.alloc_fragment((BT,BK),dtype)
                k_local1 = T.alloc_fragment((BT,BK),dtype)

                b_dq = T.alloc_fragment((BT, BK), accum_dtype)
                b_dq_shared = T.alloc_shared((BT,BK), dtype)
                dq_shared = T.alloc_shared((BT,BK), accum_dtype)
                b_dq1 = T.alloc_fragment((BT, BK), accum_dtype)
                b_dk = T.alloc_fragment((BT, BK), accum_dtype)
                b_dk_shared = T.alloc_shared((BT,BK), dtype)
                dk_shared = T.alloc_shared((BT,BK), accum_dtype)
                b_dk1 = T.alloc_fragment((BT, BK), accum_dtype)
                b_ds = T.alloc_fragment((BT, BT), accum_dtype)
                b_ds_cast = T.alloc_fragment((BT, BT), dtype)
                b_ds_shared = T.alloc_shared((BT, BT), dtype)
                b_dg_last = T.alloc_fragment((1),accum_dtype)
                b_dg_last_tmp = T.alloc_fragment((1),accum_dtype)
                b_dg_last_tmp2 = T.alloc_fragment((BK),accum_dtype)
                b_dg_last_tmp3 = T.alloc_fragment((BK),accum_dtype)
                b_dg_last_tmp_shared = T.alloc_shared((BK),accum_dtype, scope="shared")
                # b_dg_last_local = T.alloc_local((1),accum_dtype)
                # b_dg_last_shared = T.alloc_shared((1),accum_dtype, scope="shared")
                b_dg_qk = T.alloc_fragment((BT,BK),accum_dtype)
                b_dg = T.alloc_fragment((BT),accum_dtype)

                b_dkk = T.alloc_fragment((1,BT*BK), accum_dtype)
                b_dg_last_tmp4 = T.alloc_fragment((BT), accum_dtype)
                b_dg_last_tmp4_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
                b_dg_last_tmp5 = T.alloc_fragment((BT), accum_dtype)
                # b_dkksum = T.alloc_fragment((BT), accum_dtype)
                # b_dkksum_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
                # b_dkksum_T = T.alloc_fragment((1,BT), accum_dtype)

                tx = T.thread_binding(0, thread_num, thread="threadIdx.x")

                bhead = bz % head
                bb = bz // head
                bheadk = bhead // head_headk_ratio
                bheadq = bhead // head_headq_ratio

                T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g_shared)
                b_g_last[0] = b_g_shared[BT-1]

                T.clear(b_dg_last)
                T.clear(b_dg)
                T.clear(b_ds)
                T.clear(b_dq)
                T.clear(b_dk)
                for i_v in T.Pipelined(NV, num_stages=num_stages):
                    T.copy(v[bb,bhead, by*BT:(by+1)*BT, i_v*BV:(i_v+1)*BV], b_v_shared)
                    T.copy(d_o[bb,bhead, by*BT:(by+1)*BT, i_v*BV:(i_v+1)*BV], b_do_shared)

                    T.copy(dh[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, i_v*BV:(i_v+1)*BV], b_dh_shared)
                    T.copy(h[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, i_v*BV:(i_v+1)*BV], b_h_shared)

                    T.gemm(b_do_shared, b_v_shared, b_ds, transpose_A=False, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    T.copy(b_h_shared, b_h_local)
                    T.copy(b_dh_shared, b_dh_local)
                    # tilelang<0.1.5
                    # for i in T.Parallel(BK*BV):
                    #     b_hdh[0,i] = b_h_local[i//BV, i % BV] * b_dh_local[i//BV, i % BV]
                    # tl only support clear=True for sharedmemory reduce
                    # T.reduce_sum(b_hdh, b_dg_last_tmp,dim=1) # , clear=True)
                    # b_dg_last[0] += b_dg_last_tmp[0]
                    # tilelang==0.1.5
                    for i, j in T.Parallel(BK, BV):
                        b_hdh2[i, j] = b_h_local[i,j] * b_dh_local[i,j]
                    T.reduce_sum(b_hdh2, b_dg_last_tmp2, dim=1)
                    T.copy(b_dg_last_tmp2, b_dg_last_tmp_shared)
                    T.copy(b_dg_last_tmp_shared, b_dg_last_tmp3)
                    T.reduce_sum(b_dg_last_tmp3, b_dg_last_tmp, dim=0)
                    b_dg_last[0] += b_dg_last_tmp[0]
                    
                    T.gemm(b_do_shared, b_h_shared, b_dq, transpose_A=False, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(b_v_shared, b_dh_shared, b_dk, transpose_A=False, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    
                T.copy(k[bb, bheadk, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK], k_shared)
                T.copy(q[bb, bheadq, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK], q_shared)
                
                b_dg_last[0] *= T.exp(b_g_last[0])

                T.copy(b_g_shared, b_g)
                for i,j in T.Parallel(BT,BK):
                    b_dq[i,j] *= T.exp(b_g[i])
                    # qmod_bwd many place
                    b_dq[i,j] *= scale

                # reg fuse
                for i,j in T.Parallel(BT,BK):
                    b_dk[i,j] *= T.exp(-b_g[i]+b_g_last[0])
                
                
                # possible accuracy loss
                T.copy(b_g_shared, b_g1)
                T.copy(b_g_shared, b_g_T)
                for i,j in T.Parallel(BT,BT):
                    b_ds[i,j] = T.if_then_else(
                        i >= j, b_ds[i,j]*scale*T.exp(b_g1[i]-b_g_T[j]), 0
                    )
                T.copy(b_ds,b_ds_cast)
                
                # for i in T.Parallel(BT*BK):
                #     b_dkk[0,i] = b_dk[i//BK,i%BK]
                # # T.copy(k_shared, k_local1)
                # for i in T.Parallel(BT*BK):
                #     b_dkk[0,i] *= k_shared[i//BK,i%BK]
                
                # TIlelang < 0.1.5
                # layout没有以b_dk 优先 # T.copy(k_shared, k_local1) 
                # for i,j in T.Parallel(BT,BK):
                #     k_local1[i,j] = b_dk[i,j]
                # for i,j in T.Parallel(BT, BK):
                #     k_local1[i,j] *= k_shared[i, j]
                # for i in T.Parallel(BT*BK):
                #     b_dkk[0,i] = k_local1[i//BK, i%BK]
                # T.reduce_sum(b_dkk, b_dg_last_tmp, dim=1)# , clear=True)
                # Tilelang == 0.1.5
                T.copy(k_shared, k_local1)
                for i,j in T.Parallel(BT,BK):
                    k_local1[i,j] *= b_dk[i,j]
                T.reduce_sum(k_local1, b_dg_last_tmp4, dim=1)
                T.copy(b_dg_last_tmp4, b_dg_last_tmp4_shared)
                T.copy(b_dg_last_tmp4_shared, b_dg_last_tmp5)
                T.reduce_sum(b_dg_last_tmp5, b_dg_last_tmp, dim=0)
                
                b_dg_last[0] += b_dg_last_tmp[0]
                
                T.gemm(b_ds_cast,k_shared,b_dq,transpose_A=False, transpose_B=False,policy=T.GemmWarpPolicy.FullRow)
                T.copy(b_ds_cast, b_ds_shared)
                T.gemm(b_ds_shared, q_shared, b_dk, transpose_A=True,transpose_B=False,policy=T.GemmWarpPolicy.FullRow)
                # implicit cast
                # T.copy(b_dq, dq_shared)
                # T.copy(dq_shared, b_dq1)
                for i,j in T.Parallel(BT,BK):
                    b_dg_qk[i,j] = b_dq[i,j]
                T.copy(q_shared, q_local)
                T.copy(k_shared, k_local)
                # T.copy(b_dk, dk_shared)
                # T.copy(dk_shared, b_dk1)
                for i,j in T.Parallel(BT,BK):
                    b_dg_qk[i,j] = b_dg_qk[i,j] * q_local[i,j] - b_dk[i,j]*k_local[i,j]

                T.reduce_sum(b_dg_qk,b_dg,dim=1)# ,clear=True)
                
                for i in T.Parallel(BT):
                    b_dg[i] = T.if_then_else(
                        i < BT-1, b_dg[i], b_dg[i]+b_dg_last[0]
                    )
                T.copy(b_dq, b_dq_shared)
                T.copy(b_dq_shared, dq[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
                # T.copy(b_dq, dq[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
                T.copy(b_dk, b_dk_shared)
                T.copy(b_dk_shared, dk[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
                # T.copy(b_dk, dk[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
                T.copy(b_dg, b_g_shared)
                T.copy(b_g_shared, dg[bx, bb, bhead, by*BT:(by+1)*BT])
                # T.copy(b_dg, dg[bx, bb, bhead, by*BT:(by+1)*BT])

        return main
    
    if tune:
        configs = generate_config_dqkg(batch,headq,headk,head,seqlen,dim,dimv,BT, device=attn_device)
        if len(configs) == 0:
            return None
        @tilelang.autotune(
            configs=configs,
            warmup=10,
            rep=10,
        )
        @tilelang.jit(out_idx=[7,8,9,])
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel()
    else:
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel

def generate_config_dv(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT,device=H100()):
    BK_dvs = [32,64,128]# ,256]
    BV_dvs = [32,64,128] # ,256]
    num_stages_dvs = [1,2] if device.platform=="CUDA" else [1]
    num_threads_dvs = [128,256]
    # TODO: tune mi250 D=256,512 bug
    
    # H100
    MMA_ATOM_M = device.mma_primitive[0]# 64
    MMA_ATOM_TRHEADS = device.threads_per_mma # 128
    smem_cap = device.smem_cap # 232448
    reg_cap = device.reg_cap * 4 # 65536 * 4
    reg_cap_per_thread = device.register_per_thread * 4 # 255 * 4
    
    # input shape
    BK_dvs = [bk for bk in BK_dvs if D_HEAD % bk == 0]
    BV_dvs = [bv for bv in BV_dvs if D_HEADV % bv == 0]
    
    config_dv = []
    for BK_dv, BV_dv, num_stages_dv, num_threads_dv in itertools.product(BK_dvs, BV_dvs, num_stages_dvs, num_threads_dvs):
        conditions_dv = [
            num_stages_dv > D_HEAD // BK_dv,
            BT % (MMA_ATOM_M*(num_threads_dv//MMA_ATOM_TRHEADS)) != 0,
            # sharedmem_chunk_dv > smem_cap,
            # reg_chunk_dv > reg_cap and reg_chunk_dv > reg_cap_per_thread * num_threads_dv,
        ]
        if any(conditions_dv):
            continue
        config_dv.append( {
        # "BT": BT,
        "BK": BK_dv,
        "BV": BV_dv,
        "num_stages": num_stages_dv,
        "num_threads": num_threads_dv,
        })
    return config_dv

# --------------- TL_KERNEL_BWD_dv
def chunk_bwd_kernel_dv(
    batch, headq, headk, head, seqlen, dim, dimv, BT, tune=False,):
    
    def kernel_func(
        BK, BV, num_stages, num_threads
    ):
        NT = seqlen // BT
        NK = dim // BK
        NV = dimv // BV
        dtype = "bfloat16" # mi250 "float16"
        accum_dtype = "float"
        num_stages = num_stages
        thread_num = num_threads

        scale = 1.0
        assert(head % headk == 0)
        head_headk_ratio = head // headk
        assert(head % headq == 0)
        head_headq_ratio = head // headq

        @T.prim_func
        def main(
            q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
            k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
            g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
            d_o: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
            dh: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
            
            dv: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        ):
            with T.Kernel(NV, NT, batch*head, threads=thread_num) as (bx, by, bz):
                b_dv = T.alloc_fragment((BT, BV), accum_dtype)
                b_dv_shared = T.alloc_shared((BT, BV), dtype)
                b_g = T.alloc_fragment((BT), accum_dtype)
                b_g1 = T.alloc_fragment((BT), accum_dtype)
                b_g2 = T.alloc_fragment((BT), accum_dtype)
                b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
                b_g_last = T.alloc_fragment((1), accum_dtype)
                b_A = T.alloc_fragment((BT, BT), accum_dtype)
                b_A_cast = T.alloc_fragment((BT, BT), dtype)
                b_q_shared = T.alloc_shared((BT, BK), dtype)
                b_q_local = T.alloc_fragment((BT, BK), dtype)
                b_k_shared = T.alloc_shared((BT, BK), dtype)
                b_k_local = T.alloc_fragment((BT, BK), dtype)
                b_dh_shared = T.alloc_shared((BK, BV), dtype)
                b_dh_local = T.alloc_fragment((BK, BV), dtype)
                b_do_shared = T.alloc_shared((BT, BV), dtype)
                
                bhead = bz % head
                bb = bz // head
                bheadk = bhead // head_headk_ratio
                bheadq = bhead // head_headq_ratio
                
                T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g_shared)
                T.copy(b_g_shared, b_g)
                b_g_last[0] = b_g_shared[BT-1]
                
                T.clear(b_dv)
                T.clear(b_A)
                for i_k in T.Pipelined(NK, num_stages=num_stages):
                    T.copy(k[bb, bheadk, by*BT:(by+1)*BT, i_k*BK:(i_k+1)*BK], b_k_shared)
                    T.copy(dh[bb, bhead, by*dim+i_k*BK:by*dim+(i_k+1)*BK, bx*BV:(bx+1)*BV], b_dh_shared)
                    T.copy(q[bb, bheadq, by*BT:(by+1)*BT, i_k*BK:(i_k+1)*BK], b_q_shared)
                    
                    T.gemm(b_k_shared, b_dh_shared, b_dv, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(b_k_shared, b_q_shared, b_A, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i,j in T.Parallel(BT, BV):
                    b_dv[i,j] *= T.exp(-b_g[i] + b_g_last[0])
                T.copy(b_g_shared, b_g1)
                T.copy(b_g_shared, b_g2)
                for i,j in T.Parallel(BT, BT):
                    b_A[i,j] *= T.exp(-b_g1[i] + b_g2[j]) * scale
                for i,j in T.Parallel(BT,BT):
                    b_A[i,j] = T.if_then_else(
                        i <= j, b_A[i,j], 0
                    )
                
                T.copy(d_o[bb, bhead, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV], b_do_shared)
                T.copy(b_A,b_A_cast)
                T.gemm(b_A_cast, b_do_shared, b_dv, policy=T.GemmWarpPolicy.FullRow)
                T.copy(b_dv, b_dv_shared)
                T.copy(b_dv_shared, dv[bb, bhead, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
    
        return main
    
    if tune:
        configs = generate_config_dv(batch,headq,headk,head,seqlen,dim,dimv,BT, device=attn_device)
        if len(configs) == 0:
            return None
        @tilelang.autotune(
            configs=configs,
            warmup=10,
            rep=10,
        )
        @tilelang.jit(out_idx=[5,])
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel()
    else:
        def kernel(BK=None, BV=None, num_stages=None, num_threads=None):
            return kernel_func(BK,BV,num_stages,num_threads)
        
        return kernel

# # compile tilelang program

# 定义在全局作用域，确保可以被 multiprocessing 序列化
def _profiling_worker(profiler_func, kwargs, queue):
    try:
        # 在新进程中执行 profile
        result = profiler_func(**kwargs)
        
        # 获取配置结果 (result.config)
        # 注意：我们只把纯 Python 数据(dict)传回主进程，避免传递包含 CUDA 句柄的对象
        if hasattr(result, 'config'):
            ret = result.config, result.latency
        else:
            ret = result # 假设 result 本身就是配置
            
        queue.put(("success", ret))
    except Exception as e:
        # 捕获所有异常并传回主进程
        err_msg = traceback.format_exc()
        queue.put(("error", err_msg))
    
def tune(tune_file, kernel_profiler, problem_keys)->Tuple:
    tuned_config = None
    tuned_latency = -1
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
                tuned_latency = config['tuned_latency']
                return tuned_config, tuned_latency
    if tuned_config is None:
        print("tune: ", problem_keys)
        # TODO: use a seperate process for autotune to avoid cuda context crash
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        p = ctx.Process(target=_profiling_worker, args=(kernel_profiler, problem_keys, queue))
        p.start()
        p.join()
        if not queue.empty():
            status, data = queue.get()
            if status == "success":
                if data is not None: # maybe no valid config found
                    tuned_config, tuned_latency = data
            else:
                raise RuntimeError(f"Profiling process failed with error:\n{data}")
        else:
            # 子进程可能因为段错误(Segfault)直接崩溃，没有向 queue 放数据
            exit_code = p.exitcode
            raise RuntimeError(f"Profiling process crashed silently (Exit code: {exit_code}). This usually indicates a Segmentation Fault or CUDA Illegal Access.")
        # result = kernel_profiler(
        #     **problem_keys
        # )
        # if result is not None:
        #     tuned_config = result.config
        #     tuned_latency = result.latency
        with open(tune_file, "w") as f:
            configs.append({
                **pk,
                'tuned_config': tuned_config,
                'tuned_latency': tuned_latency
            })
            json.dump(configs, f, indent=4)
    return tuned_config, tuned_latency
    
def get_problem_keys_ho(BT):
    return {
        "batch": {{BATCH}},
        "headq": {{HQ}},
        "headk": {{HK}},
        "head": {{H}},
        "seqlen": {{N_CTX}},
        "dim": {{D_HEAD}},
        "dimv": {{D_HEADV}},
        "BT": BT,
    }
    
def autotune_linearattn(file_path="mamba2"):
            
    BTs = [32,64,128] # ,192,256]
 
    best_config_h, best_config_o = {}, {}
    best_latency = 1e6
    best_BT = None
    for BT in BTs:
        config_h, latency_h = tune(f"{file_path}_h.json", partial(chunk_fwd_h, tune=True), get_problem_keys_ho(BT))
        config_o, latency_o = tune(f"{file_path}_o.json", partial(chunk_o, tune=True), get_problem_keys_ho(BT))
        
        if latency_h == -1 or latency_o == -1:
            continue
        if latency_h + latency_o < best_latency:
            best_BT = BT
            best_latency = latency_h + latency_o
            best_config_h = {
                "BK": config_h["BK"],
                "BV": config_h["BV"],
                "num_stages": config_h["num_stages"],
                "num_threads": config_h["num_threads"],
            }
            best_config_o = {
                "BK": config_o["BK"],
                "BV": config_o["BV"],
                "num_stages": config_o["num_stages"],
                "num_threads": config_o["num_threads"],
            }
    
    return best_BT, best_config_h, best_config_o, best_latency


def autotune_linearattn_bwd(file_path="mamba2"):
    
    BTs = [32,64,128] # ,192]# ,256]
 
    best_config_dh, best_config_dqkg, best_config_dv = {}, {}, {}
    best_latency = 1e6
    best_BT = None
    for BT in BTs:
        config_h, latency_h = tune(f"{file_path}_h.json", partial(chunk_fwd_h, tune=True), get_problem_keys_ho(BT))
        config_dh, latency_dh = tune(f"{file_path}_dh.json", partial(chunk_bwd_kernel_dh, tune=True), get_problem_keys_ho(BT))
        config_dqkg, latency_dqkg = tune(f"{file_path}_dqkg.json", partial(chunk_bwd_dqkg, tune=True), get_problem_keys_ho(BT))
        config_dv, latency_dv = tune(f"{file_path}_dv.json", partial(chunk_bwd_kernel_dv, tune=True), get_problem_keys_ho(BT))
        
        if latency_h == -1 or latency_dh == -1 or latency_dqkg == -1 or latency_dv == -1:
            continue
        if latency_h + latency_dh + latency_dqkg + latency_dv < best_latency:
            best_BT = BT
            best_latency = latency_h + latency_dh + latency_dqkg + latency_dv
            best_config_h = {
                "BK": config_h["BK"],
                "BV": config_h["BV"],
                "num_stages": config_h["num_stages"],
                "num_threads": config_h["num_threads"],
            }
            best_config_dh = {
                "BK": config_dh["BK"],
                "BV": config_dh["BV"],
                "num_stages": config_dh["num_stages"],
                "num_threads": config_dh["num_threads"],
            }
            best_config_dqkg = {
                "BK": config_dqkg["BK"],
                "BV": config_dqkg["BV"],
                "num_stages": config_dqkg["num_stages"],
                "num_threads": config_dqkg["num_threads"],
            }
            best_config_dv = {
                "BK": config_dv["BK"],
                "BV": config_dv["BV"],
                "num_stages": config_dv["num_stages"],
                "num_threads": config_dv["num_threads"],
            }
    
    return best_BT, best_config_h, best_config_dh, best_config_dqkg, best_config_dv, best_latency


BT = None
BT_BWD = None
chunk_fwd_h_mod = None
chunk_fwd_o_mod = None
chunk_fwd_h_mod_2 = None
chunk_bwd_dh_mod = None
chunk_bwd_dqkg_mod = None
chunk_bwd_dv_mod = None

# 初始化标志位
_IS_INITIALIZED = False

def initialize_module():
    global _IS_INITIALIZED
    if _IS_INITIALIZED:
        return
    
    global BT, BT_BWD
    global chunk_fwd_h_mod, chunk_fwd_o_mod
    global chunk_fwd_h_mod_2, chunk_bwd_dh_mod, chunk_bwd_dqkg_mod, chunk_bwd_dv_mod
    
    TUNE = {{TUNE}}
    TUNE_FILE = "{{TUNE_FILE}}"
    TUNE_BWD = {{TUNE_BWD}}
    TUNE_FILE_BWD = "{{TUNE_FILE_BWD}}"

    tuned_config_h = None
    tuned_config_o = None
    BT=None
    if TUNE:
        BT, tuned_config_h, tuned_config_o, _ = autotune_linearattn(TUNE_FILE)
    else:
        BT = {{BT}}
        tuned_config_h = {
            'BK': {{BK_h}},
            'BV': {{BV_h}},
            'num_stages': {{num_stages_h}},
            'num_threads': {{num_threads_h}}
        }
        tuned_config_o = {
            'BK': {{BK_o}},
            'BV': {{BV_o}},
            'num_stages': {{num_stages_o}},
            'num_threads': {{num_threads_o}}
        }
        
        
    BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV = {{BATCH}}, {{HQ}}, {{HK}}, {{H}}, {{N_CTX}}, {{D_HEAD}}, {{D_HEADV}}

    chunk_fwd_h_mod = tl.compile(chunk_fwd_h(BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT)(**tuned_config_h), {{output_idx_list_h}})
    output_idx_list = {{output_idx_list_o}}# [5,]
    chunk_fwd_o_mod = tl.compile(chunk_o(BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT)(**tuned_config_o), output_idx_list, )

    # bwd
    BT_BWD=None
    if TUNE_BWD:
        BT_BWD, tuned_config_h_2, tuned_config_dh, tuned_config_dqkg, tuned_config_dv,_ = autotune_linearattn_bwd(TUNE_FILE_BWD)
    else:
        BT_BWD = {{BT_BWD}}
        tuned_config_h_2 = {
            'BK': {{BK_h}},
            'BV': {{BV_h}},
            'num_stages': {{num_stages_h}},
            'num_threads': {{num_threads_h}}
        }
        tuned_config_dh = {
            'BK': {{BK_dh}},
            'BV': {{BV_dh}},
            'num_stages': {{num_stages_dh}},
            'num_threads': {{num_threads_dh}}
        }
        tuned_config_dqkg = {
            'BK': {{BK_dqkg}},
            'BV': {{BV_dqkg}},
            'num_stages': {{num_stages_dqkg}},
            'num_threads': {{num_threads_dqkg}}
        }
        tuned_config_dv = {
            'BK': {{BK_dv}},
            'BV': {{BV_dv}},
            'num_stages': {{num_stages_dv}},
            'num_threads': {{num_threads_dv}}
        }
    chunk_fwd_h_mod_2 = tl.compile(chunk_fwd_h(BATCH, HQ,HK, H, N_CTX, D_HEAD, D_HEADV, BT_BWD)(**tuned_config_h_2), {{output_idx_list_h}})
    chunk_bwd_dh_mod = tl.compile(chunk_bwd_kernel_dh(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT_BWD)(**tuned_config_dh), [5,])
    chunk_bwd_dqkg_mod = tl.compile(chunk_bwd_dqkg(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT_BWD)( **tuned_config_dqkg), [7,8,9,])
    chunk_bwd_dv_mod = tl.compile(chunk_bwd_kernel_dv(BATCH, HQ, HK, H, N_CTX, D_HEAD, D_HEADV, BT_BWD)( **tuned_config_dv), [5,])

    _IS_INITIALIZED = True

# --------------- TL_INTERFACE
class LinearAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, decay, {{custom_inputs_list}} ):
        BATCH, HQ, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        HK = k.shape[1]
        H = v.shape[1]
        
        if not _IS_INITIALIZED:
            initialize_module()

        # decay_mod here
        {{decay_mod_expr | indent(8)}}

        # k_mod here
        {{k_mod_expr | indent(8)}}

        # v_mod here
        {{v_mod_expr | indent(8)}}

        global BT
        decay_cumsum = chunk_local_cumsum_scalar(
            {{decay_name}}, BT
        )
        global chunk_fwd_h_mod, chunk_fwd_o_mod

        h = chunk_fwd_h_mod({{k_name}}, {{v_name}}, decay_cumsum, {{custom_inputs_list_h}})
        o = chunk_fwd_o_mod(h, {{q_name}}, {{k_name}}, {{v_name}}, decay_cumsum, {{custom_inputs_list_o}})

        ctx.save_for_backward(q, k, v, decay, {{custom_inputs_list}} ) # , decay_cumsum
        # ctx.BT = BT
        return o

    @staticmethod
    def backward(ctx, d_o):
        if not _IS_INITIALIZED:
            initialize_module()
        d_o = d_o.contiguous()
        global BT_BWD
        BT2 = BT_BWD
        q, k, v, decay, {{custom_inputs_list}} = ctx.saved_tensors #  decay_cumsum

        BATCH, HQ, N_CTX, D_HEAD = q.shape
        D_HEADV = v.shape[-1]
        HK = k.shape[1]
        H = v.shape[1]
        NT = N_CTX // BT2
        
        # decay_mod here
        {{decay_mod_expr1 | indent(8)}}

        # k_mod here
        {{k_mod_expr1 | indent(8)}}

        # v_mod here
        {{v_mod_expr1 | indent(8)}}
        
        # q_mod here
        {{q_mod_expr1 | indent(8)}}
        
        decay_cumsum = chunk_local_cumsum_scalar(
            {{decay_name2}}, BT2
        )
        
        global chunk_fwd_h_mod_2, chunk_bwd_dh_mod, chunk_bwd_dqkg_mod, chunk_bwd_dv_mod
        
        h = chunk_fwd_h_mod_2({{k_name2}}, {{v_name2}}, decay_cumsum, {{custom_inputs_list_h}})
        
        # k_mod 2 (2 not the same time as 1)
        {{k_mod_expr_2 | indent(8)}}
        # v_mod 2
        {{v_mod_expr_2 | indent(8)}}
        
        dh = chunk_bwd_dh_mod({{q_name1}}, {{k_name1}}, {{v_name1}}, decay_cumsum, d_o)
        dq, dk, ddecay = chunk_bwd_dqkg_mod({{q_name1}}, {{k_name1}}, {{v_name1}}, h, decay_cumsum, d_o, dh)
        if HQ < H:
            dq =  dq.view(BATCH, HQ, H//HQ, N_CTX, D_HEAD).sum(2)
        if HK < H:
            dk = dk.view(BATCH, HK, H//HK, N_CTX, D_HEAD).sum(2)
        ddecay = ddecay.sum(0)
        dg2 = torch.empty(ddecay.shape, dtype=torch.float32, device=ddecay.device)
        compute_final_dg[(NT, BATCH*H)](ddecay, dg2, T=N_CTX, BT=BT2)
        dv = chunk_bwd_dv_mod({{q_name1}}, {{k_name1}}, decay_cumsum, d_o, dh)
        
        # v_bwd
        {{v_mod_bwd_expr | indent(8)}}
        # decay_mod_bwd
        {{decay_mod_bwd_expr | indent(8)}}
        # k_bwd
        {{k_mod_bwd_expr | indent(8)}}
        # q_bwd
        {{q_mod_bwd_expr | indent(8)}}
        
        return {{dq_name}}, {{dk_name}}, {{dv_name}}, {{ddecay_name}}, {{custom_inputs_grad_list}}
        
        

linear_attention = LinearAttention.apply
