# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from functools import partial
import torch
import torch.nn.functional as F
import tilelang
# from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum
import argparse
import os
import json
from typing import Tuple

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

import itertools
def get_configs():
    num_splits = [1, 2, 4, 8]
    block_Ns = [32, 64, 128] if attn_device.platform == "CUDA" else [16, 32, 64]
    block_Ms = [64] if attn_device.platform == "CUDA" else [16, 32, 64]
    stages = [2,] if attn_device.platform == "CUDA" else [0,]
    shared_fuse = [True,] if attn_device.platform == "CUDA" else [False,]
    _configs = list(itertools.product(num_splits, block_Ns, block_Ms, stages, shared_fuse))
    configs = [
        {
            "block_N": c[1], 
            "block_H": c[2],
            "num_split": c[0],
            "num_stages": c[3],
            "shared_fuse": c[4],
        } for c in _configs
    ]
    return configs

def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, tune=False):
    
    def kernel_func(block_N, block_H, num_split, num_stages, shared_fuse):
        scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
        dtype = "{{tl_dtype}}"
        accum_dtype = "float"
        kv_group_num = heads // kv_head_num
        VALID_BLOCK_H = min(block_H, kv_group_num)
        assert kv_head_num == 1, "kv_head_num must be 1"

        @T.macro
        def flash_attn(
                Q: T.Tensor([batch, 1, heads, dim], dtype),
                Q_pe: T.Tensor([batch, 1, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                Output: T.Tensor([batch, 1, heads, dim], dtype),
        ):
            with T.Kernel(batch, heads // min(block_H, kv_group_num), threads=256) as (bx, by):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                Q_local = T.alloc_fragment([block_H, dim], dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
                Q_pe_local = T.alloc_fragment([block_H, pe_dim], dtype)
                KV_shared = T.alloc_shared([block_N, dim], dtype)
                K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared = T.alloc_shared([block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                cur_kv_head = by // (kv_group_num // block_H)
                T.use_swizzle(10)
                T.annotate_layout({
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                })

                if shared_fuse:
                    T.copy(Q[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_shared)
                    T.copy(Q_pe[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
                else:
                    T.copy(Q[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_local)
                    T.copy(Q_pe[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_local)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(seqlen_kv, block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(KV[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], KV_shared)
                    T.copy(K_pe[bx, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                    T.clear(acc_s)
                    if shared_fuse:
                        T.gemm(
                            Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                        T.gemm(
                            Q_pe_shared,
                            K_pe_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol)
                    else:
                        T.gemm(
                            Q_local, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.gemm(
                            Q_pe_local,
                            K_pe_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    if shared_fuse:
                        T.copy(acc_s, S_shared)
                    else:
                        T.copy(acc_s, acc_s_cast)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    if shared_fuse:
                        T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                    else:
                        T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                if shared_fuse:
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])
                else:
                    T.copy(acc_o, Output[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :])

        @T.macro
        def flash_attn_split(
                Q: T.Tensor([batch, 1, heads, dim], dtype),
                Q_pe: T.Tensor([batch, 1, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split, 1], dtype),
                Output_partial: T.Tensor([batch, 1, heads, num_split, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // min(block_H, kv_group_num), num_split, threads=256) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                Q_local = T.alloc_fragment([block_H, dim], dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
                Q_pe_local = T.alloc_fragment([block_H, pe_dim], dtype)
                KV_shared = T.alloc_shared([block_N, dim], dtype)
                K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared = T.alloc_shared([block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                cur_kv_head = by // (kv_group_num // block_H)
                T.use_swizzle(10)
                T.annotate_layout({
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                    S_shared: tilelang.layout.make_swizzled_layout(S_shared),
                })

                if shared_fuse:
                    T.copy(Q[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_shared)
                    T.copy(Q_pe[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
                else:
                    T.copy(Q[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_local)
                    T.copy(Q_pe[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_local)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    kv_start = (seqlen_kv // num_split) * bz + k * block_N
                    kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                    T.copy(KV[bx, kv_start:kv_end, cur_kv_head, :], KV_shared)
                    T.copy(K_pe[bx, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                    T.clear(acc_s)
                    if shared_fuse:
                        T.gemm(
                            Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                        T.gemm(
                            Q_pe_shared,
                            K_pe_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol)
                    else:
                        T.gemm(
                            Q_local, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        T.gemm(
                            Q_pe_local,
                            K_pe_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    if shared_fuse:
                        T.copy(acc_s, S_shared)
                        T.copy(S_shared, acc_s_cast)
                    else:
                        T.copy(acc_s, acc_s_cast)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    if shared_fuse:
                        T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                    else:
                        T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                for i in T.Parallel(block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, glse[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz, 0])
                if shared_fuse:
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output_partial[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz, :])
                else:
                    T.copy(acc_o, Output_partial[bx, 0, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split, 1], dtype),
                Output_partial: T.Tensor([batch, 1, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, 1, heads, dim], dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                po_local = T.alloc_fragment([dim], dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_local([1], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.annotate_layout({
                    lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                lse_max_local[0] = -T.infinity(accum_dtype)
                for k in T.serial(num_split):
                    lse_max_local[0] = T.max(lse_max_local[0], glse[bz, by, k, 0])
                for k in T.Pipelined(num_split, num_stages=1):
                    lse_local_split[0] = glse[bz, by, k, 0]
                    lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial[bz, 0, by, k, i]
                    lse_local_split[0] = glse[bz, by, k, 0]
                    scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[bz, 0, by, i] = o_accum_local[i]

        @T.prim_func
        def main_split(
                Q: T.Tensor([batch, 1, heads, dim], dtype),
                Q_pe: T.Tensor([batch, 1, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split, 1], dtype),
                Output_partial: T.Tensor([batch, 1, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, 1, heads, dim], dtype),
        ):
            flash_attn_split(Q, Q_pe, KV, K_pe, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def main_no_split(
                Q: T.Tensor([batch, 1, heads, dim], dtype),
                Q_pe: T.Tensor([batch, 1, heads, pe_dim], dtype),
                KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
                K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
                glse: T.Tensor([batch, heads, num_split, 1], dtype),
                Output_partial: T.Tensor([batch, 1, heads, num_split, dim], dtype),
                Output: T.Tensor([batch, 1, heads, dim], dtype),
        ):
            flash_attn(Q, Q_pe, KV, K_pe, Output)

        if num_split > 1:
            return main_split
        else:
            return main_no_split
        
    if tune:
        configs = get_configs()
        @tilelang.autotune(
            configs=configs,
            warmup=10,
            rep=10,
        )
        @tilelang.jit(out_idx=[6,])
        def kernel(block_N=None, block_H=None, num_split=None, num_stages=None, shared_fuse=None):
            return kernel_func(block_N, block_H, num_split, num_stages, shared_fuse)

        return kernel()
    else:
        def kernel(block_N=None, block_H=None, num_split=None, num_stages=None, shared_fuse=None):
            return kernel_func(block_N, block_H, num_split, num_stages, shared_fuse)

        return kernel

def ref_program(q, q_pe, kv, k_pe, glse, Output_partial):
    #     """
    #     Inputs:
    #     - q (Tensor): [batch, heads, dim]
    #     - q_pe (Tensor): [batch, heads, pe_dim]
    #     - kv (Tensor): [batch, seqlen_kv, kv_head_num, dim]
    #     - k_pe (Tensor): [batch, seqlen_kv, kv_head_num, pe_dim]
    #     - glse (Tensor): [batch, heads, num_split]
    #     - Output_partial (Tensor): [batch, heads, num_split, dim]
    #     Outputs:
    #     - output (Tensor): [batch, heads, dim]
    #     """
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    num_head_groups = q.shape[1] // kv.shape[2]
    scale = (dim + pe_dim)**0.5
    q = rearrange(
        q, 'b (h g) d -> b g h d', g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    q_pe = rearrange(
        q_pe, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

    kv = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

    k_pe = rearrange(k_pe, 'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, pe_dim]

    query = torch.concat([q, q_pe], dim=-1)
    key = torch.concat([kv, k_pe], dim=-1)

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, kv,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out

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
        result = kernel_profiler(
            **problem_keys
        )
        if result is not None:
            tuned_config = result.config
            tuned_latency = result.latency
        with open(tune_file, "w") as f:
            configs.append({
                **pk,
                'tuned_config': tuned_config,
                'tuned_latency': tuned_latency
            })
            json.dump(configs, f, indent=4)
    return tuned_config, tuned_latency
 

TUNE=True
BLOCK_N = 64
BLOCK_H = 64
num_split = 1
num_stages = 2 if attn_device.platform == "CUDA" else 0
shared_fuse = True if attn_device.platform == "CUDA" else False

if TUNE:
    problem_keys = {
        "batch": {{BATCH}},
        "heads": {{HEADS}},
        "kv_head_num": {{KV_HEAD_NUM}},
        "seqlen_kv": {{KV_CTX}},
        "dim": {{DIM}},
        "pe_dim": {{PE_DIM}},
    }
    tuned_config, tuned_latency = tune(f"tuned_config/{attn_device.name}/mla_decode.json", partial(flashattn, tune=True), problem_keys)
    BLOCK_N = tuned_config["block_N"]
    BLOCK_H = tuned_config["block_H"]
    num_split = tuned_config["num_split"]
    num_stages = tuned_config["num_stages"]
    shared_fuse = tuned_config["shared_fuse"]

program = flashattn(
    {{BATCH}}, {{HEADS}}, {{KV_HEAD_NUM}}, {{KV_CTX}},
    {{DIM}}, {{PE_DIM}})(**tuned_config)

mod = tilelang.compile(program, out_idx=[6])

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, q_pe, kv, k_pe):
        global num_split
        glse = torch.empty(
            (q.shape[0], q.shape[2], num_split, q.shape[1]), dtype=q.dtype, device=q.device)
        Output_partial = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], num_split, kv.shape[-1]), dtype=q.dtype, device=q.device)
        o = mod(q, q_pe, kv, k_pe, glse, Output_partial)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        pass
    
attention = _attention.apply

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--heads', type=int, default=128, help='q heads number')
    parser.add_argument('--kv_heads', type=int, default=1, help='kv heads number')
    parser.add_argument('--kv_ctx', type=int, default=8192, help='kv context length')
    parser.add_argument('--dim', type=int, default=512, help='head dim')
    parser.add_argument('--pe_dim', type=int, default=64, help='pe head dim')
    args = parser.parse_args()
    batch, heads, kv_heads, kv_ctx, dim, pe_dim = args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim
    qk_flops = 2 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2 * batch * heads * kv_ctx * dim
    total_flops = qk_flops + pv_flops
    BLOCK_N = 64
    BLOCK_H = 64
    num_split = 1

    program = flashattn(batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split)
    kernel = tilelang.compile(program, out_idx=[6])
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    latency = profiler.do_bench(warmup=500)
    print(f"Latency: {latency} ms")
    print(f"TFlops: {total_flops / latency * 1e-9} TFlops")