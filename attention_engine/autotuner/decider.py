from core.utils import meta_tensor
from typing import Tuple

import itertools


def next_multiple_of(x, y):
    return x + (y - x) % y


def memory_usage(block_M, block_N, block_K, block_KV,
                 qk_mem_level, acco_mem_level,
                 num_threads, stages, dtype):
    """
    calculate memory usage
    """
    dtype_size = dtype.itemsize
    dtype_accum_size = 4
    bytes_per_register = 4
    shared_mem = 0

    shared_mem += block_M * block_K * dtype_size + \
        block_N * block_K * stages * dtype_size + \
        block_M * block_N * qk_mem_level[1] * dtype_accum_size

    shared_mem += block_N * block_KV * stages * dtype_size + \
        block_M * block_KV * acco_mem_level[1] * dtype_accum_size

    # store do
    # shared_mem = shared_mem - block_N*block_K*stages*dtype_size + max(block_N*block_K*stages*dtype_size, block_M*block_KV*dtype_size)

    reg_num = 0
    reg_num += block_M * block_N * qk_mem_level[0] * dtype_accum_size
    reg_num += block_M * block_KV * acco_mem_level[0] * dtype_accum_size

    reg_num /= (num_threads * bytes_per_register)
    return shared_mem, reg_num


def decider(qkv_meta, hardware_meta) -> Tuple[bool, dict]:

    batch, seqlen_q, head_q, dim_qk = qkv_meta[0].shape
    head_k = qkv_meta[1].shape[2]
    seqlen_kv = qkv_meta[1].shape[1]
    dim_v = qkv_meta[2].shape[3]
    dtype = qkv_meta[0].dtype

    # shared memory tile
    block_M_stride = hardware_meta.mma_primitive[0]
    block_M_max = next_multiple_of(seqlen_q, block_M_stride)
    block_M = [
        bm for bm in range(
            block_M_stride,
            block_M_max + 1,
            block_M_stride)]

    block_N_stride = hardware_meta.mma_primitive[1]
    block_N_max = next_multiple_of(seqlen_kv, block_N_stride)
    block_N = [
        bn for bn in range(
            block_N_stride,
            block_N_max + 1,
            block_N_stride)]

    block_K_stride = hardware_meta.mma_primitive[2]
    block_K_max = next_multiple_of(dim_qk, block_K_stride)
    # block_K = [bk for bk in range(block_K_stride, block_K_max + 1, block_K_stride)]
    # divided by dim_qk
    block_K = [
        bk for bk in range(
            block_K_stride,
            block_K_max +
            1,
            block_K_stride) if dim_qk %
        bk == 0]

    num_threads_stride = hardware_meta.threads_per_mma
    num_threads_max = hardware_meta.threads_cap
    num_threads = [
        nt for nt in range(
            num_threads_stride,
            num_threads_max + 1,
            num_threads_stride)]

    qk_mem_level = [[1, 0, 0], [1, 1, 0]]  # ,[1,1,1]]
    acco_mem_level = [[1, 0, 0], [1, 1, 0]]

    MAX_STAGE = 4
    stages = [s for s in range(1, MAX_STAGE + 1)]

    config_iter = itertools.product(
        block_M, block_N, block_K, qk_mem_level, acco_mem_level, num_threads, stages
    )
    configs = []
    for bm, bn, bk, qk_mem, acco_mem, nt, stage in config_iter:
        shared_mem, reg_num = memory_usage(
            bm, bn, bk, dim_v, qk_mem, acco_mem, nt, stage, dtype)
        if shared_mem <= hardware_meta.smem_cap and reg_num * \
                nt <= hardware_meta.reg_cap and reg_num <= hardware_meta.register_per_thread:

            # register fuse, warp row
            if bm % ((nt / hardware_meta.threads_per_mma)
                     * hardware_meta.mma_primitive[0]):
                continue
            # implememnt block_K==dim_qk
            if bk != dim_qk:
                continue

            # how to model?
            # data reuse : block_M larger
            # pipeline: acco_mem_level 0, acc_s_memlevel 0
            #

            configs.append({
                "block_M": bm,
                "block_N": bn,
                "block_K": bk,
                "qk_mem_level": qk_mem,
                "acco_mem_level": acco_mem,
                "num_threads": nt,
                "stages": stage,
                "shared_mem": shared_mem,
                "reg_num": reg_num
            })

    need_fuse = len(configs) > 0

    return need_fuse, configs


if __name__ == "__main__":
    import torch
    qkv_meta = [
        meta_tensor((1, 2048, 12, 96), dtype=torch.bfloat16),
        meta_tensor((1, 2048, 12, 96), dtype=torch.bfloat16),
        meta_tensor((1, 2048, 12, 192), dtype=torch.bfloat16),
    ]
    from autotuner.arch import H100
    hardware_meta = H100()
    need_fuse, fuse_configs = decider(qkv_meta, hardware_meta)
    # print(fuse_configs)
    for config in fuse_configs:
        print(config)
