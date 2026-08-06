from examples.mha import causal_softmax_attention
from examples.mha_v2 import causal_softmax_attention as causal_softmax_attention_v2
from examples.mha_decode import softmax_attention_decode
from examples.gated_retention import gated_retention
from examples.sigmoid_attn import sigmoid_attention
from examples.sigmoid_attn_v2 import sigmoid_attention as sigmoid_attention_v2
from examples.reluattn import relu_attention
from examples.reluattn_v2 import relu_attention as relu_attention_v2
from examples.retnet_recurrent import retnet_recurrent
from examples.retention_parallel import retention_parallel
from examples.mamba2 import mamba2
from examples.mla_decode_v2 import mla_decode
from examples.sparse_gqa_decode import sparse_gqa_decode

from plot_fig_h100 import plot_figure11

# from attention_engine.benchmark.bench_utils import do_bench
from tilelang.profiler import do_bench

import torch
import torch.nn.functional as F
import math
import triton
import pandas as pd
import os
from einops import rearrange, einsum, repeat
from functools import lru_cache
from typing import Optional

import time
from termcolor import cprint

import torch._functorch.config

# Disable the "donated buffer" to enable the use of "retain_graph=True" in torch.compile
torch._functorch.config.donated_buffer = False

RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)

def log_section(title):
    print("\n" + "="*60)
    cprint(f" {title}", "magenta", attrs=["bold"])
    print("="*60)

def log_success(msg):
    cprint(f" ✔ {msg}", "green")


def bench_fig11():
    
    Batches = [1,8]
    seqlens = [2048, 4096, 8192]
    
    # (a) Softmax Attention (DeepSeek-V2-Lite)
    log_section("(a) Softmax Attention (DeepSeek-V2-Lite)")
    deepseek_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        torch.cuda.empty_cache()
        result_dict = bench_attention("causal_softmax_attn", b, 16, s, s, 192, 128)
        deepseek_data.append((f"BS{b}\nS{s}", result_dict))
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("causal_softmax_attn", b, 16, 1, s, 192, 128, require_grad=False)
        deepseek_data.append((f"BS{b}S1\nKV{s}", result_dict))
    dump_bench_result("deepseek", deepseek_data)
        
    # (b) Softmax Attention (LLAMA-3.1-8B)
    log_section("(b) Softmax Attention (LLAMA-3.1-8B)")
    llama_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("causal_softmax_attn", b, 32, s, s, 128, 128)
        llama_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("llama", llama_data)
        
    # (c) ReLU Attention (ViT-s/16-style)
    log_section("(c) ReLU Attention (ViT-s/16-style)")
    vit_data = []
    for b, s in [(B, S) for B in [32,64] for S in [512, 1024, 2048]]:
        result_dict = bench_attention("relu_attn", b, 6, s, s, 64, 64)
        vit_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("vit", vit_data)
        
    # (d) Softmax Attention (Diff-Transformer-3B)
    log_section("(d) Softmax Attention (Diff-Transformer-3B)")
    dit_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("causal_softmax_attn", b, 12, s, s, 128, 256)
        dit_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("dit", dit_data)
        
    # (e) Retention Parallel (RetNet-6.7B)
    log_section("(e) Retention Parallel (RetNet-6.7B)")
    retnet_data = []
    for b, s in [(B, S) for B in Batches for S in [2048, 4096]]:
        torch.cuda.empty_cache()
        result_dict = bench_attention("retention_parallel", b, 32, s, s, 256, 512)
        retnet_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("retnet", retnet_data)
    
    # (f) Sigmoid Attention (LLAMA-3.1-8B)
    log_section("(f) Sigmoid Attention (LLAMA-3.1-8B)")
    sigmoid_attn_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("sigmoid_attn", b, 32, s, s, 128, 128)
        sigmoid_attn_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("sigmoid_attn", sigmoid_attn_data)
    
    # (h) Gated Retention (RFA-Big)
    log_section("(h) Gated Retention (RFA-Big)")
    rfa_data = []
    for b, s in [(B, S) for B in [64,] for S in [1024, 2048, 4096]]:
        result_dict = bench_attention("gated_retention", b, 16, s, s, 64, 64)
        rfa_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("rfa", rfa_data)
    
    # (g) Mamba2 SSM (Mamba2-2.7B)
    log_section("(g) Mamba2 SSM (Mamba2-2.7B)")
    mamba2_data = []
    for b, s in [(B, S) for B in Batches for S in seqlens]:
        result_dict = bench_attention("mamba2_ssm", b, 1, s, s, 128, 64, head_v=80)
        mamba2_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("mamba2", mamba2_data)
    
    # (i) Gated Retention (YOCO-13B)
    log_section("(i) Gated Retention (YOCO-13B)")
    yoco_data = []
    for b, s in [(B, S) for B in [8,] for S in [1024, 2048, 4096]]:
        torch.cuda.empty_cache()
        result_dict = bench_attention("gated_retention", b, 40, s, s, 256, 256)
        yoco_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("yoco", yoco_data)
        
        
    # (j) RetNet Recurrent (RetNet-6.7B)
    log_section("(j) RetNet Recurrent (RetNet-6.7B)")
    retnet_recur_data = []
    for b, s in [(B, S) for B in Batches for S in [2048, 4096]]:
        torch.cuda.empty_cache()
        result_dict = bench_attention("retention_recurrent", b, 32, s, s, 256, 512)
        retnet_recur_data.append((f"BS{b}\nS{s}", result_dict))
    dump_bench_result("retnet_recur", retnet_recur_data)

    # (k) DeepSeek MLA
    log_section("(k) DeepSeek MLA")
    mla_data = []
    for b, s in [(B, S) for B in [8,] for S in seqlens]:
        result_dict = bench_attention("mla_attn", b, 128, 1, s, 576, 512, head_k=1, head_v=1)
        mla_data.append((f"BS{b}S1\nKV{s}", result_dict))
    dump_bench_result("mla", mla_data)
    

    # (l) Sparse Group Query Attention
    log_section("(l) Sparse Group Query Attention")
    sparse_gqa_data = []
    for b, s in [(B, S) for B in [8,] for S in seqlens]:
        result_dict = bench_attention("sparse_gqa", b, 32, 1, s, 128, 128, head_k=8, head_v=8)
        sparse_gqa_data.append((f"BS{b}S1\nKV{s}", result_dict))
    dump_bench_result("sparse_gqa", sparse_gqa_data)
    
    
def dump_bench_result(name: str, data: list):
    """
    export benchmark data to fwd and bwd CSV files.
    Rows are method names (e.g., MetaAttention), columns are configurations (e.g., BS1\\nS2048).
    """
    print(name, data)
    if not data:
        return
    
    # 1. Prepare containers
    # We use dictionaries to build data, structure: { method_name: { config_name: value } }
    fwd_data_dict = {}
    bwd_data_dict = {}
    
    # To maintain the order of columns (i.e., the order of configurations appearing in data)
    config_order = []

    # 2. Parse data
    for config_name, metrics in data:
        if config_name not in config_order:
            config_order.append(config_name)
            
        if not metrics:
            continue

        for method_name, values in metrics.items():
            if method_name not in fwd_data_dict:
                fwd_data_dict[method_name] = {}
            if method_name not in bwd_data_dict:
                bwd_data_dict[method_name] = {}
            

            if values is None:
                fwd_val, bwd_val = None, None
            else:
                # Safe unpacking to prevent tuple length issues
                fwd_val = values[0] if len(values) > 0 else None
                bwd_val = values[1] if len(values) > 1 else None
            
            fwd_data_dict[method_name][config_name] = fwd_val
            bwd_data_dict[method_name][config_name] = bwd_val

    # 3. Convert to DataFrame
    # orient='index' means dictionary keys (Method Name) are used as row indices
    df_fwd = pd.DataFrame.from_dict(fwd_data_dict, orient='index')
    df_bwd = pd.DataFrame.from_dict(bwd_data_dict, orient='index')

    # 4. Reorder columns
    # Ensure the column order in the CSV matches the order in the input data list
    # Some configurations may exist in data but be missing in some methods, pandas will automatically fill NaN
    # Take the intersection here to prevent duplicates in data or missing columns in DataFrame
    valid_cols = [c for c in config_order if c in df_fwd.columns]
    df_fwd = df_fwd[valid_cols]
    
    valid_cols_bwd = [c for c in config_order if c in df_bwd.columns]
    df_bwd = df_bwd[valid_cols_bwd]

    df_fwd.columns = df_fwd.columns.astype(str).str.replace('\n', ' ')
    df_bwd.columns = df_bwd.columns.astype(str).str.replace('\n', ' ')

    fwd_filename = os.path.join(RESULT_DIR, f"{name}_fwd.csv")
    bwd_filename = os.path.join(RESULT_DIR, f"{name}_bwd.csv")

    df_fwd.to_csv(fwd_filename, index_label="Method")
    df_bwd.to_csv(bwd_filename, index_label="Method")

    print(f"Saved: {fwd_filename}")
    print(f"Saved: {bwd_filename}")

def bench_attention(attn_type:str, Batch:int, head:int, seqlen_q:int, seqlen_kv:int, dim_qk:int, dim_v:int, head_k: int=None, head_v: int=None, require_grad: bool=True):
    if head_k is None:
        head_k = head
    if head_v is None:
        head_v = head
        
    result_dict = {}
    if attn_type == "causal_softmax_attn":
        result_dict = bench_softmaxattention(Batch, head, seqlen_q, seqlen_kv, dim_qk, dim_v, require_grad=require_grad)
    elif attn_type == "sigmoid_attn":
        result_dict = bench_sigmoidattention(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "gated_retention":
        result_dict = bench_gated_retention(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "relu_attn":
        result_dict = bench_reluattention(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "retention_parallel":
        result_dict = bench_retention_parallel(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "retention_recurrent":
        result_dict = bench_retnet_recurrent(Batch, head, seqlen_q, dim_qk, dim_v)
    elif attn_type == "mamba2_ssm":
        result_dict = bench_mamba2_ssm(Batch, head, seqlen_q, dim_qk, dim_v, HK=head_k, HV=head_v)
    elif attn_type == "mla_attn":
        result_dict = bench_mla_decode(Batch, head, seqlen_kv, dim_qk, dim_v, HKV=head_k)
    elif attn_type == "sparse_gqa":
        result_dict = bench_sparse_gqa_decode(Batch, head, head_k, seqlen_kv, dim_qk, dim_v)
    else:
        # raise ValueError(f"Undefined attention type: {attn_type}")
        print(f"Warning: Undefined attention type {attn_type}, skipping benchmark.")
        result_dict = {}

    return result_dict

    
def bench_softmaxattention(B, H, Sq, S, D, DV, device='cuda', dtype=torch.float16, require_grad=True):
    
    # init input
    query = torch.randn(B, Sq, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, Sq, H, DV, device=device, dtype=dtype, requires_grad=False)

    result_dict = {}
    
    # ours
    if Sq < S:
        # decode
        assert require_grad == False
        attention_module = softmax_attention_decode(B, H, Sq, S, D, DV)
    else:
        attention_module = causal_softmax_attention(B, H, S, D, DV, tune=True)
    def ours():
        o = attention_module(query, key, value)
        return o
    ours_fwd_lat = do_bench(ours)
    if require_grad:
        o = attention_module(query, key, value)
        ours_bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    else:
        ours_bwd_lat = None
    
    attention_module_v2 = causal_softmax_attention_v2(B, H, S, D, DV)
    def ours_v2():
        o = attention_module_v2(query, key, value)
        return o
    ours_fwd_lat_v2 = do_bench(ours_v2)
    if require_grad:
        o = attention_module_v2(query, key, value)
        ours_bwd_lat_v2 = do_bench(lambda: o.backward(do, retain_graph=True))
    else:
        ours_bwd_lat_v2 = None
    ours_fwd_lat = min(ours_fwd_lat, ours_fwd_lat_v2)
    if require_grad:
        ours_bwd_lat = min(ours_bwd_lat, ours_bwd_lat_v2)
        
    result_dict["MetaAttention"] = (ours_fwd_lat, ours_bwd_lat)
    
    # FlashAttention-2
    fa2_lat = None
    try:
        from flash_attn import flash_attn_func
        def fa2(dim_padded):
            if D < dim_padded:
                query_padded = F.pad(query, (0, dim_padded - D), value=0.)
                key_padded = F.pad(key, (0, dim_padded - D), value=0.)
            else:
                query_padded = query
                key_padded = key
            if DV < dim_padded:
                value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
            else:
                value_padded = value
            o_ref = flash_attn_func(
                query_padded,
                key_padded,
                value_padded,
                softmax_scale=(
                    1 / D)**0.5,
                causal=True)
            if DV < dim_padded:
                o_ref = o_ref[:, :, :, :DV]
            return o_ref
        
        dim_padded_fa2 = max(D, DV)
        fa2_fwd_lat = do_bench(lambda: fa2(dim_padded_fa2))
        if require_grad:
            o_ref = fa2(dim_padded_fa2)
            fa2_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        else:
            fa2_bwd_lat = None
        result_dict["FlashAttention-2"] = (fa2_fwd_lat, fa2_bwd_lat)
    except Exception as e:
        print(f"Warning: FlashAttention-2 not available: {e}")

    # FlashAttention-3
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_func_hopper
        
        def fa3(dim_padded=0):
            if D < dim_padded:
                query_padded = F.pad(query, (0, dim_padded - D), value=0.)
                key_padded = F.pad(key, (0, dim_padded - D), value=0.)
            else:
                query_padded = query
                key_padded = key
            if DV < dim_padded:
                value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
            else:
                value_padded = value
            o_ref = flash_attn_func_hopper(
                query_padded, key_padded, value_padded, softmax_scale=(
                    1 / D)**0.5, causal=True)
            if DV < dim_padded:
                o_ref = o_ref[:, :, :, :DV]
            return o_ref
        
        dim_padded_fa3 = list(filter(lambda x: x >= max(D, DV), [64, 128, 192, 256]))
        assert len(dim_padded_fa3) > 0, "No valid padding size for FlashAttention-3"
        dim_padded_fa3 = min(dim_padded_fa3)
        # flash attention 3 specifically supported for D=192 and DV=128, so does not need padding for this case
        if D == 192 and DV == 128:
            dim_padded_fa3 = 0
        
        fa3_fwd_lat = do_bench(lambda: fa3(dim_padded_fa3))
        
        if require_grad:
            o_ref = fa3(dim_padded_fa3)
            fa3_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        else:
            fa3_bwd_lat = None
        
        result_dict["FlashAttention-3"] = (fa3_fwd_lat, fa3_bwd_lat)
        
    except Exception as e:
        print(f"Warning: FlashAttention-3 not available: {e}")
        
    # FLex Attention
    if Sq == S:
        try:
            query2 = query.transpose(1, 2).contiguous()
            key2 = key.transpose(1, 2).contiguous()
            value2 = value.transpose(1, 2).contiguous()
            do2 = do.transpose(1, 2).contiguous()
            query2.detach_().requires_grad_(require_grad)
            key2.detach_().requires_grad_(require_grad)
            value2.detach_().requires_grad_(require_grad)
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
            
            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            
            @lru_cache
            def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
                block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
                return block_mask

            block_mask = create_block_mask_cached(
                causal_mask, 1, 1, S, S, device=query.device)
            flex_attention = torch.compile(flex_attention)
            
            dim_padded_flex = list(filter(lambda x: x >= max(D, DV), [64, 128, 256]))
            assert (len(dim_padded_flex) > 0)
            dim_padded_flex = min(dim_padded_flex)

            def flex_ref(dim_padded):
                if D < dim_padded:
                    query_padded = F.pad(query2, (0, dim_padded - D), value=0.)
                    key_padded = F.pad(key2, (0, dim_padded - D), value=0.)
                else:
                    query_padded = query2
                    key_padded = key2
                if DV < dim_padded:
                    value_padded = F.pad(value2, (0, dim_padded - DV), value=0.)
                else:
                    value_padded = value2
                o_ref = flex_attention(
                    query_padded,
                    key_padded,
                    value_padded,
                    block_mask=block_mask)
                if DV < dim_padded:
                    o_ref = o_ref[:, :, :, :DV]
                return o_ref
            
            flex_fwd_lat = do_bench(lambda: flex_ref(dim_padded_flex))
            if require_grad:
                o_ref = flex_ref(dim_padded_flex)
                flex_bwd_lat = do_bench(lambda: o_ref.backward(do2, retain_graph=True))
            else:
                flex_bwd_lat = None
            result_dict["FlexAttention"] = (flex_fwd_lat, flex_bwd_lat)
        except Exception as e:
            print(f"Warning: Flex Attention not available: {e}")
        
    # torch inductor
    try:
        query.detach_().requires_grad_(require_grad)
        key.detach_().requires_grad_(require_grad)
        value.detach_().requires_grad_(require_grad)
        if Sq == 1:
            causal = False
        else:
            causal = True
        @torch.compile
        def ref(query, key, value, causal=True, softmax_scale=None):
            dim = query.shape[-1]
            num_head_groups = query.shape[2] // key.shape[2]
            if softmax_scale is None:
                softmax_scale = 1 / dim** 0.5

            query = rearrange(
                query, 'b s (h g) d -> b s g h d',
                g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]
            scores = einsum(query, key,
            'b s g h d, b t h d -> b g h s t')
            if causal:
                seqlenq = query.shape[1]
                seqlenk = key.shape[1]
                mask = torch.tril(
                    torch.ones(
                        seqlenq, seqlenk, device=scores.device))
                mask = mask.unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attention = F.softmax(
                scores * softmax_scale, dim=-1)

            out = einsum(attention, value,
                    'b g h s t, b t h d -> b g h s d')
            out = rearrange(out, 'b g h s d -> b s (h g) d') 
            return out
        
        ref_fwd_lat = do_bench(lambda: ref(query, key, value, causal=causal))
        if require_grad:
            o_ref = ref(query, key, value, causal=causal)
            ref_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        else:
            ref_bwd_lat = None
            
        result_dict["Torch Inductor"] = (ref_fwd_lat, ref_bwd_lat)
    
    except Exception as e:
        print(f"Warning: Torch Inductor benchmark failed to compile ref: {e}")
        
        
    return result_dict

def bench_sigmoidattention(B, H, S, D, DV, dtype=torch.float16, require_grad=True):
    
    result_dict = {}
    
    accum_dtype = torch.float32
    query = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=False)
    
    softmax_bias = 0.1 * torch.randn(1, device="cuda", dtype=accum_dtype, requires_grad=False)
    
    softmax_bias_2 = softmax_bias.to("cpu")
    
    # ours
    attention_module = sigmoid_attention(B, H, S, D, DV, tune=True)
    attention_module_v2 = sigmoid_attention_v2(B, H, S, D, DV)
    
    query1 = query.clone().detach().requires_grad_(False)
    key1 = key.clone().detach().requires_grad_(False)
    value1 = value.clone().detach().requires_grad_(False)
    
    softmax_bias_1 = softmax_bias.clone().detach().requires_grad_(False)
    fwd_lat = do_bench(lambda: attention_module(query1, key1, value1, softmax_bias_1))
    if require_grad:
        o = attention_module(query, key, value, softmax_bias)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    fwd_lat_v2 = do_bench(lambda: attention_module_v2(query1, key1, value1, softmax_bias_1))
    fwd_lat = min(fwd_lat, fwd_lat_v2)
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # flash-sigmoid
    try:
        from flash_sigmoid import flash_attn_func
        
        fwd_lat_ref = do_bench(lambda: flash_attn_func(
            query,
            key,
            value,
            softmax_scale=1.0,
            causal=True,
            sigmoid_bias=softmax_bias_2))
        if require_grad:
            out_ref = flash_attn_func(
                query,
                key,
                value,
                softmax_scale=1.0,
                causal=True,
                sigmoid_bias=softmax_bias_2)
            bwd_lat_ref = do_bench(lambda: out_ref.backward(do, retain_graph=True))
            
        result_dict["FlashSigmoid"] = (fwd_lat_ref, bwd_lat_ref)
    except Exception as e:
        print(f"Warning: flash-sigmoid not available: {e}")
        
    # pytorch
    try:
        @torch.compile
        def ref(query, key, value, causal=True, softmax_scale=None):
            dim = query.shape[-1]
            num_head_groups = query.shape[2] // key.shape[2]
            if softmax_scale is None:
                softmax_scale = 1 / dim** 0.5

            query = rearrange(
                query, 'b s (h g) d -> b s g h d',
                g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]
            scores = einsum(query, key,
            'b s g h d, b t h d -> b g h s t')
            if causal:
                seqlenq = query.shape[1]
                seqlenk = key.shape[1]
                mask = torch.tril(
                    torch.ones(
                        seqlenq, seqlenk, device=scores.device))
                mask = mask.unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attention = F.softmax(
                scores * softmax_scale, dim=-1)

            out = einsum(attention, value,
                    'b g h s t, b t h d -> b g h s d')
            out = rearrange(out, 'b g h s d -> b s (h g) d') 
            return out
    
        torch_fwd_lat = do_bench(lambda: ref(query, key, value))
        if require_grad:
            o_ref = ref(query, key, value)
            torch_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        result_dict["Torch Inductor"] = (torch_fwd_lat, torch_bwd_lat)
    except Exception as e:
        print(f"Warning: Torch Inductor benchmark failed to compile ref: {e}")
    return result_dict
    
def bench_reluattention(B, H, S, D, DV, device='cuda', dtype=torch.float16, require_grad=True):
    
    result_dict = {}
    query = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=False)
    
    # ours
    attention_module = relu_attention(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = do_bench(lambda: attention_module(query, key, value))
    if require_grad:
        o = attention_module(query, key, value)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    attention_module_v2 = relu_attention_v2(B, H, S, D, DV, dtype=dtype)
    fwd_lat_v2 = do_bench(lambda: attention_module_v2(query, key, value))
    fwd_lat = min(fwd_lat, fwd_lat_v2)
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # Pytorch ReLU Attention
    def ref_program(query, key, value):
        qk = torch.einsum('bqhd,bkhd->bhqk', query, key)
        qk = qk / (D ** 0.5)
        qk = F.relu(qk)
        o = torch.einsum('bhqk,bkhd->bqhd', qk, value)
        return o

    ref_program_fwd_lat = do_bench(lambda: ref_program(query, key, value))
    if require_grad:
        out_ref = ref_program(query, key, value)
        ref_program_bwd_lat = do_bench(lambda: out_ref.backward(do, retain_graph=True))
    result_dict["Torch Inductor"] = (ref_program_fwd_lat, ref_program_bwd_lat)
    
    return result_dict

def bench_gated_retention(B, H, S, D, DV, device='cuda', dtype=torch.bfloat16, require_grad=True):
    
    result_dict = {}
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    g = F.logsigmoid(torch.randn(B, H, S, device="cuda", dtype=accum_dtype)).clamp_min(-5)
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    
    q.detach_().requires_grad_(require_grad)
    k.detach_().requires_grad_(require_grad)
    g.detach_().requires_grad_(require_grad)
    v.detach_().requires_grad_(require_grad)


    q1 = q.clone()
    k1 = k.clone()
    v1 = v.clone()
    g1 = g.clone().to(dtype)
    
    q1.detach_().requires_grad_(require_grad)
    k1.detach_().requires_grad_(require_grad)
    g1.detach_().requires_grad_(require_grad)
    v1.detach_().requires_grad_(require_grad)
    
    # ours
    attention_module = gated_retention(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = do_bench(lambda: attention_module(q, k, v, g))
    if require_grad:
        o = attention_module(q, k, v, g)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # flash-linear-attention
    try:
        from fla.ops.simple_gla import chunk_simple_gla
        fwd_lat_ref = do_bench(lambda: chunk_simple_gla(
            q1, k1, v1, g1, head_first=True
        )[0])
        
        if require_grad:
            out_ref,_ = chunk_simple_gla(
                q1, k1, v1, g1, head_first=True
            )
            bwd_lat_ref = do_bench(lambda: out_ref.backward(do, retain_graph=True))
        else:
            bwd_lat_ref = None
        result_dict["FlashLinearAttention"] = (fwd_lat_ref, bwd_lat_ref)

    except Exception as e:
        print(f"Warning: fla.ops.simple_gla not available: {e}")
        
    # torch inductor
    try:
        from fla.ops.simple_gla.naive import torch_simple_gla
        q1.detach_().requires_grad_(require_grad)
        k1.detach_().requires_grad_(require_grad)
        v1.detach_().requires_grad_(require_grad)
        g1.detach_().requires_grad_(require_grad)
        
        torch_simple_gla = torch.compile(torch_simple_gla)
        fwd_lat_ref2 = do_bench(lambda: torch_simple_gla(
            q1, k1, v1, g1, chunk_size=512
        ))
        
        if require_grad:
            out_ref2 = torch_simple_gla(
                q1, k1, v1, g1, chunk_size=512
            )
            bwd_lat_ref2 = do_bench(lambda: out_ref2.backward(do, retain_graph=True))
        else:
            bwd_lat_ref2 = None
        result_dict["Torch Inductor"] = (fwd_lat_ref2, bwd_lat_ref2)
    except Exception as e:
        print(f"Warning: Torch Inductor benchmark failed: {e}")
        

    return result_dict
   
def bench_retnet_recurrent(B, H, S, D, DV, device="cuda", dtype=torch.bfloat16, require_grad=True):
    
    result_dict = {}
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    g = torch.tensor(range(0, H), dtype=accum_dtype)
    g = 1 - torch.exp2(-5 - g)
    g = g[None, :, None].expand(B, H, S).cuda().detach().contiguous()
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    
    q.detach_().requires_grad_(require_grad)
    k.detach_().requires_grad_(require_grad)
    g.detach_().requires_grad_(False)
    v.detach_().requires_grad_(require_grad)

    # clone for reference
    q1 = q.clone()
    k1 = k.clone()
    v1 = v.clone()
    g1 = g.clone()

    q1.detach_().requires_grad_(require_grad)
    k1.detach_().requires_grad_(require_grad)
    g1.detach_().requires_grad_(False)
    v1.detach_().requires_grad_(require_grad)

    # ours
    attention_module = retnet_recurrent(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = do_bench(lambda: attention_module(q, k, v, g))
    if require_grad:
        o = attention_module(q, k, v, g)
        bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
    
    # flash-linear-attention
    try:
        from fla.ops.retention import chunk_retention
        fwd_lat_ref = do_bench(lambda: chunk_retention(
            q1, k1, v1, head_first=True
        )[0])
        if require_grad:
            o_ref, _ = chunk_retention(
                q1, k1, v1, head_first=True
            )
            bwd_lat_ref = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        result_dict["FlashLinearAttention"] = (fwd_lat_ref, bwd_lat_ref)
    except Exception as e:
        print(f"Warning: fla.ops.retention not available: {e}")
        
    # torch inductor
    try:
        @torch.compile
        def ref(q, k, v):
            orig_type = q.dtype
            q, k, v = q.float(), k.float(), v.float()
            _, n_heads, seq_len, d_head = q.shape
            s = (1 - q.new_tensor(2., dtype=torch.float).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
            n = q.new_tensor(range(seq_len), dtype=torch.float)
            n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
            s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
            o = torch.einsum('bhqk,bhkd->bhqd', s, v)
            return o.to(orig_type)
        
        fwd_lat_ref2 = do_bench(lambda: ref(q1, k1, v1))
        if require_grad:
            o_ref2 = ref(q1, k1, v1)
            bwd_lat_ref2 = do_bench(lambda: o_ref2.backward(do, retain_graph=True))
        result_dict["Torch Inductor"] = (fwd_lat_ref2, bwd_lat_ref2)
    except Exception as e:
        print(f"Warning: Torch Inductor benchmark failed: {e}")
                
    
    return result_dict

def bench_retention_parallel(B, H, S, D, DV, device="cuda", dtype=torch.float16, require_grad=False):
    
    result_dict = {}
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=require_grad)
    v = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, S, H, DV, device="cuda", dtype=dtype, requires_grad=False)
    mask = torch.rand(
        1, H, S, S, device="cuda", dtype=dtype, requires_grad=False
    ).tril().contiguous()


    # ours
    attention_module = retention_parallel(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = do_bench(lambda: attention_module(q, k, v, mask))
    
    result_dict["MetaAttention"] = (fwd_lat, None)
    
    # pytorch 
    try:
        @torch.compile
        def ref_program(q, k, v, mask):
            qk = torch.einsum('bqhd,bkhd->bhqk', q, k)
            qkm = qk * mask
            r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
            o = torch.einsum('bhqk,bkhd->bqhd', qkm / r, v)
            return o.to(dtype=dtype)

        ref_lat = do_bench(lambda: ref_program(q, k, v, mask))
        result_dict["Torch Inductor"] = (ref_lat, None)
    except Exception as e:
        print(f"Warning: Pytorch Retention benchmark failed: {e}")
    
    return result_dict

def bench_mamba2_ssm(B, HQ, S, D, DV, HK=None, HV=None, dtype=torch.bfloat16, require_grad=True):
    
    result_dict = {}
    
    # init input
    query = torch.randn(B, S, HQ, D, device="cuda", dtype=dtype)
    key = torch.randn(B, S, HK, D, device="cuda", dtype=dtype)
    value = torch.randn(B, S, HV, DV, device="cuda", dtype=dtype)
    if require_grad:
        do = 0.1 * torch.randn(B, S, HV,
                                     DV, dtype=dtype, device="cuda")
    A_mamba = 1.5 * torch.randn(HV, dtype=dtype, device="cuda") - 4.0
    # initialize dt
    accum_dtype = torch.float32
    dt_mamba = 0.7 * torch.randn(B, S, HV, dtype=accum_dtype, device="cuda")
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    dt_min = 0.001
    dt_max = 0.1
    dt = torch.exp(
        torch.rand(HV, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)
    
    # ours
    q_ours = query.clone().transpose(1, 2).contiguous()
    k_ours = key.clone().transpose(1, 2).contiguous()
    v_ours = value.clone().transpose(1, 2).contiguous()
    A_ours = A_mamba[None, :].clone().contiguous()
    dt_ours = dt_mamba.clone().transpose(1, 2).contiguous()
    if require_grad:
        do_ours = do.clone().transpose(1, 2).contiguous()
    
    q_ours = q_ours.detach().requires_grad_(require_grad)
    k_ours = k_ours.detach().requires_grad_(require_grad)
    v_ours = v_ours.detach().requires_grad_(require_grad)
    A_ours = A_ours.detach().requires_grad_(require_grad)
    dt_ours = dt_ours.detach().requires_grad_(require_grad)
    
    attention_module = mamba2(B, HQ, S, D, DV, HK, HV, dtype=dtype, tune=True)
    fwd_lat = do_bench(
        lambda: attention_module(q_ours, k_ours, v_ours, dt_ours, A_ours, dt_ours.to(dtype))
    )
    if require_grad:
        o = attention_module(q_ours, k_ours, v_ours, dt_ours, A_ours, dt_ours.to(dtype))
        bwd_lat = do_bench(lambda: o.backward(do_ours, retain_graph=True))
    result_dict["MetaAttention"] = (fwd_lat, bwd_lat)
        
    # mamba2 ssm
    value = value.detach_().requires_grad_(require_grad)
    A_mamba.detach_().requires_grad_(require_grad)
    dt_mamba.detach_().requires_grad_(require_grad)
    key.detach_().requires_grad_(require_grad)
    query.detach_().requires_grad_(require_grad)

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        fwd_lat_ref = do_bench(
            lambda: mamba_chunk_scan_combined(
                value,
                dt_mamba,
                A_mamba,
                key,
                query,
                chunk_size=64,
            )
        )
        if require_grad:
            out_ref = mamba_chunk_scan_combined(
                value,
                dt_mamba,
                A_mamba,
                key,
                query,
                chunk_size=64,
            )
            bwd_lat_ref = do_bench(lambda: out_ref.backward(do, retain_graph=True))
        else:
            bwd_lat_ref = None
        result_dict["Mamba2"] = (fwd_lat_ref, bwd_lat_ref)
    except Exception as e:
        print(f"Warning: mamba2 ssm not available: {e}")
        
    # torch inductor
    try:
        def chunk_state_ref(B, x, dt, dA_cumsum):
            """
            Argument:
                B: (batch, seqlen, ngroups, headdim)
                x: (batch, seqlen, nheads, headdim)
                dt: (batch, nheads, nchunks, chunk_size)
                dA_cumsum: (batch, nheads, nchunks, chunk_size)
            Return:
                states: (batch, nchunks, nheads, headdim, dstate)
            """
            # Check constraints.
            batch, seqlen, nheads, headdim = x.shape
            dstate = B.shape[-1]
            _, _, nchunks, chunk_size = dt.shape
            assert seqlen <= nchunks * chunk_size
            assert x.shape == (batch, seqlen, nheads, headdim)
            assert dt.shape == (batch, nheads, nchunks, chunk_size)
            ngroups = B.shape[2]
            assert nheads % ngroups == 0
            assert B.shape == (batch, seqlen, ngroups, dstate)
            B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
            assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
            if seqlen < nchunks * chunk_size:
                x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
                B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
            x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
            B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
            decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
            return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)

        def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
            """
            Argument:
                states: (batch, nchunks, nheads, dim)
                dA_chunk_cumsum: (batch, nheads, nchunks)
                initial_states: (batch, nheads, dim)
            Return:
                out: (batch, nchunks, nheads, dim)
                final_states: (batch, nheads, dim)
            """
            if initial_states is None:
                initial_states = torch.zeros_like(states[:, 0])
            states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
            dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
            dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
            nchunks = dA_chunk_cumsum.shape[-1]
            # (batch, nheads, nchunks, nchunks)
            dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
            # (batch, nheads, nchunks, nchunks)
            decay_chunk = torch.exp(dt_chunk_segment_sum)
            causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
            decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
            out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
            return out[:, :-1], out[:, -1]

        def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
            """
            Argument:
                B: (batch, seqlen, ngroups, dstate)
                C: (batch, seqlen, ngroups, dstate)
                x: (batch, seqlen, nheads, headdim)
                dt: (batch, nheads, nchunks, chunk_size)
                dA_cumsum: (batch, nheads, nchunks, chunk_size)
                prev_states: (batch, nchunks, nheads, headdim, dstate)
                D: (nheads, headdim) or (nheads,)
                z: (batch, seqlen, nheads, headdim)
            Return:
                out: (batch, seqlen, nheads, headdim)
            """
            batch, seqlen, nheads, headdim = x.shape
            _, _, ngroups, dstate = B.shape
            assert B.shape == (batch, seqlen, ngroups, dstate)
            _, _, nchunks, chunk_size = dt.shape
            assert seqlen == nchunks * chunk_size
            assert C.shape == B.shape
            B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
            C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
            CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
            # (batch, nheads, nchunks, chunksize, chunksize)
            dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
            decay = torch.exp(dt_segment_sum)
            scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
            causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
            scores_decay = scores_decay.masked_fill(~causal_mask, 0)
            out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                            rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
            state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
            out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                                    prev_states.to(C.dtype)) * state_decay_out
            out = out + out_prev
            out = rearrange(out, "b c l h p -> b (c l) h p")
            if D is not None:
                if D.dim() == 1:
                    D = rearrange(D, "h -> h 1")
                out = out + x * D
            return out if z is None else out * F.silu(z)
        
        @torch.compile
        def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
            """
            Argument:
                x: (batch, seqlen, nheads, headdim)
                dt: (batch, seqlen, nheads)
                A: (nheads)
                B: (batch, seqlen, ngroups, dstate)
                C: (batch, seqlen, ngroups, dstate)
                D: (nheads, headdim) or (nheads,)
                z: (batch, seqlen, nheads, headdim)
                dt_bias: (nheads,)
            Return:
                out: (batch, seqlen, nheads, headdim)
            """
            batch, seqlen, nheads, headdim = x.shape
            dstate = B.shape[-1]
            if seqlen % chunk_size != 0:
                dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
            dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
            dt = dt.float()  # We want high precision for this before cumsum
            if dt_bias is not None:
                dt = dt + rearrange(dt_bias, "h -> h 1 1")
            if dt_softplus:
                dt = F.softplus(dt)
            dA = dt * rearrange(A, "h -> h 1 1")
            dA_cumsum = torch.cumsum(dA, dim=-1)
            # 1. Compute the state for each chunk
            states = chunk_state_ref(B, x, dt, dA_cumsum)
            states_dtype = states.dtype
            if states.dtype not in [torch.float32, torch.float64]:
                states = states.to(torch.float32)
            # 2. Pass the state to all the chunks by weighted cumsum.
            # state_passing_ref is much less numerically stable
            states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                            "... (p n) -> ... p n", n=dstate)
            states = states.to(states_dtype)
            # 3. Compute the output for each chunk
            out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
            return out

        fwd_lat_ref2 = do_bench(
            lambda: ssd_chunk_scan_combined_ref(
                value, 
                dt_mamba,
                A_mamba,
                key,
                query,
                chunk_size=64,
            )
        )
        if require_grad:
            out_ref2 = ssd_chunk_scan_combined_ref(
                value, 
                dt_mamba,
                A_mamba,
                key,
                query,
                chunk_size=64,
            )
            bwd_lat_ref2 = do_bench(lambda: out_ref2.backward(do, retain_graph=True))
        else:
            bwd_lat_ref2 = None
        result_dict["Torch Inductor"] = (fwd_lat_ref2, bwd_lat_ref2)
    except Exception as e:
        print(f"Warning: Torch Inductor benchmark failed: {e}")
    return result_dict

def bench_mla_decode(B, HQ, SKV, D, DV, HKV=1, dtype=torch.bfloat16):
    
    result_dict = {}

    q = torch.randn(B, 1, HQ, D, dtype=dtype, device="cuda")
    # To be compatible with flashMLA
    KV = torch.randn(B*SKV//64,64, HKV, DV, dtype=dtype, device="cuda")
    k_pe = torch.randn(B*SKV//64,64, HKV, D-DV, dtype=dtype, device="cuda")
    KV = torch.concat([KV, k_pe], dim=-1).contiguous()
    
    # ours
    attention_module = mla_decode(B, HQ, SKV, D, DV, HK=HKV, HV=HKV, dtype=dtype)
    fwd_lat = do_bench(lambda: attention_module(q, KV))
    result_dict["MetaAttention"] = (fwd_lat, None)
    
    # flashMLA
    try:
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
        
        cache_seqlens = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens, 1 * HQ // HKV, HKV
        )
        block_size = 64
        block_table = torch.arange(
            B * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda"
        ).view(B, max_seqlen_pad // block_size)
        
        fwd_lat_ref = do_bench(lambda: flash_mla_with_kvcache(
            q,
            KV,
            block_table,
            cache_seqlens,
            DV,
            tile_scheduler_metadata,
            num_splits,
            causal=True)
        )

        result_dict["FlashMLA"] = (fwd_lat_ref, None)

    except Exception as e:
        print(f"Warning: flashMLA not available: {e}")
        
    # triton
    try:
        from ref.flash_mla_decode_triton import flash_mla_triton
        
        q_nope, q_pe = q[..., :DV].contiguous(), q[..., DV:].contiguous()
        blocked_k_nope, blocked_k_pe = KV[..., :DV].contiguous(), KV[..., DV:].contiguous()
        cache_seqlens = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
        max_seqlen = cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        
        block_size = 64
        block_table = torch.arange(
            B * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda"
        ).view(B, max_seqlen_pad // block_size)
        
        fwd_lat_ref = do_bench(lambda: flash_mla_triton(
            q_nope, q_pe,
            block_table,
            blocked_k_nope, blocked_k_pe,
            max_seqlen_pad,
            block_size,
            B, 1, cache_seqlens, HQ, HKV, D, DV, True, dtype)
        )
        
        result_dict["FlashMLA Triton"] = (fwd_lat_ref, None)
    
    except Exception as e:
        print(f"Warning: Triton MLA Decode not available: {e}")
        
    return result_dict

def bench_sparse_gqa_decode(B, HQ, HKV, SKV, D, DV, dtype=torch.float16):
    
    result_dict = {}
    
    block_size = 32
    sparse_ratio = 0.8
    
    q = torch.randn(B, 1, HQ, D, dtype=dtype, device="cuda")
    key = torch.randn(B, SKV, HKV, D, dtype=dtype, device="cuda")
    value = torch.randn(B, SKV, HKV, DV, dtype=dtype, device="cuda")
    cache_seqlens = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
    
    def generate_block_mask(batch, heads_kv, max_cache_seqlen, sparse_ratio, cache_seqlens):
        num_blocks = (max_cache_seqlen + block_size - 1) // block_size

        valid_num_blocks = torch.ceil(cache_seqlens * (1 - sparse_ratio) / block_size).int()
        # print("valid_num_blocks: ", valid_num_blocks)
        max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
        # print("max_valid_num_blocks: ", max_valid_num_blocks)
        # Initialize block_mask with false (for padding blocks)
        block_mask = torch.zeros((batch, heads_kv, num_blocks), dtype=torch.bool, device='cuda')

        # Assign valid indices while ensuring no duplicates within each batch-group
        for b in range(batch):
            max_valid_block = max_valid_num_blocks[b].item()  # Max valid blocks for this batch
            valid_num_block = valid_num_blocks[b].item()  # Valid blocks for this batch
            if valid_num_block > 0:  # Ensure there's at least one valid block
                for h in range(heads_kv):
                    perm = torch.randperm(max_valid_block, device='cuda')[:valid_num_block]
                    block_mask[b, h, perm] = True
                    
        return block_mask
    
    block_mask = generate_block_mask(B, HKV, SKV, sparse_ratio, cache_seqlens)
    
    # ours
    attention_module = sparse_gqa_decode(B, HQ, HKV, SKV, D, DV, dtype=dtype, BLOCK=block_size)
    fwd_lat = do_bench(lambda: attention_module(q, key, value, block_mask=block_mask, cache_seqlens=cache_seqlens), warmup=100)
    
    result_dict["MetaAttention"] = (fwd_lat, None)
    
    # triton ref
    try:
        from ref.sparse_gqa_decode_varlen_triton import block_sparse_flash_decode_gqa_mask_triton
        
        fwd_lat_ref = do_bench(lambda: block_sparse_flash_decode_gqa_mask_triton(
            q, key, value, cache_seqlens, SKV, block_mask, block_size), warmup=100
        )
        
        result_dict["SeerAttention"] = (fwd_lat_ref, None)
    except Exception as e:
        print(f"Warning: Triton Sparse GQA not available: {e}")
        
    # torch inductor
    try:
        sparse_mask = torch.zeros(B, HQ//HKV, HKV, SKV, dtype=torch.bool, device='cuda')
        # Assign mask values
        for b in range(B):
            for h in range(HKV):
                for idx in range((SKV + block_size - 1) // block_size):
                    if block_mask[b, h, idx]:
                        sparse_mask[b, :, h, idx * block_size:(idx + 1) * block_size] = 1
        @torch.compile
        def ref_program_torch(query, key, value, sparse_mask, cache_seqlens, max_cache_seqlen, num_blocks,
                      block_size):
            query = query.squeeze(1)  # [batch_size, heads, dim]
            batch, heads, dim = query.shape
            heads_kv = key.shape[2]

            num_head_groups = query.shape[1] // key.shape[2]
            scale = dim**0.5
            key = rearrange(key, 'b n h d -> b h n d')  # [batch_size, heads_kv, seqlen_kv, dim]
            value = rearrange(value, 'b n h d -> b h n d')  # [batch_size, heads_kv, seqlen_kv, dim]

            query = rearrange(
                query, 'b (h g) d -> b g h d',
                g=num_head_groups)  # [batch_size, num_head_groups, heads_kv, dim]

            scores = einsum(
                query, key,
                'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

            # sparse_mask = torch.zeros_like(scores)
            # # Assign mask values
            # for b in range(batch):
            #     for h in range(heads_kv):
            #         for idx in range(num_blocks):
            #             if block_mask[b, h, idx]:
            #                 sparse_mask[b, :, h, idx * block_size:(idx + 1) * block_size] = 1

            scores = scores.masked_fill(sparse_mask == 0, float('-inf'))

            range_len = torch.arange(scores.shape[-1], device='cuda').unsqueeze(0)
            cache_seqlens_expanded = cache_seqlens.unsqueeze(1)
            pad_mask = range_len >= cache_seqlens_expanded
            pad_mask = pad_mask[:, None, None, :]
            scores = scores.masked_fill(pad_mask, float('-inf'))
            attention = F.softmax(
                scores / scale, dim=-1)  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

            out = einsum(attention, value,
                        'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, heads_kv, dim]
            out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
            return out
        
        fwd_lat_ref2 = do_bench(lambda: ref_program_torch(
            q, key, value, sparse_mask, cache_seqlens, SKV,
            (SKV + block_size - 1) // block_size, block_size
        ))
        
        result_dict["Torch Inductor"] = (fwd_lat_ref2, None)
    except Exception as e:
        print(f"Warning: Torch Inductor benchmark failed: {e}")

    
    return result_dict



if __name__ == "__main__":
    print("\n" + "#"*60)
    cprint("        STARTING BENCHMARK (FIGURE 11 - H100)", "green", attrs=["bold", "reverse"])
    print("#"*60 + "\n")
    
    start_time = time.time()
    bench_fig11()
    elapsed = time.time() - start_time
    
    print("\n" + "#"*60)
    cprint(f"        BENCHMARK COMPLETED IN {elapsed:.2f} SECONDS", "green", attrs=["bold", "reverse"])
    print("#"*60 + "\n")

    plot_figure11(RESULT_DIR, "figure11_h100.pdf")
    log_success(f"Figure 11 plotted and saved to figure11_h100.pdf")
    

    
