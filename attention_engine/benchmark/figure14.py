from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from termcolor import cprint

from examples.mamba2 import mamba2
from examples.mha import causal_softmax_attention
from examples.mha_decode import softmax_attention_decode
from examples.mla_decode import mla_decode
from examples.reluattn import relu_attention
from examples.retnet_recurrent import retnet_recurrent

from .cases import mi250_cases
from .results import dump_bench_result


def log_section(title: str) -> None:
    print("\n" + "=" * 60)
    cprint(f" {title}", "magenta", attrs=["bold"])
    print("=" * 60)


def log_success(message: str) -> None:
    cprint(f" [OK] {message}", "green")


def _resolve_timer(timer: Callable[..., float] | None) -> Callable[..., float]:
    if timer is not None:
        return timer
    from tilelang.profiler import do_bench

    return do_bench


def bench_fig12(
    result_dir: str | Path = "results_mi250",
    timer: Callable[..., float] | None = None,
) -> tuple[Path, ...]:
    """Benchmark and write the exact 25-point Figure 14 matrix."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    cases = mi250_cases()
    output_paths: list[Path] = []
    for dataset in dict.fromkeys(case.dataset for case in cases):
        dataset_cases = [case for case in cases if case.dataset == dataset]
        data = []
        log_section(dataset)
        for case in dataset_cases:
            result_dict = bench_attention(
                case.attn_type,
                case.batch,
                case.heads,
                case.seqlen_q,
                case.seqlen_kv,
                case.dim_qk,
                case.dim_v,
                head_k=case.head_k,
                head_v=case.head_v,
                dtype=getattr(torch, case.dtype),
                require_grad=case.require_grad,
                chunk_size=case.chunk_size,
                block_size=case.block_size,
                sparse_ratio=case.sparse_ratio,
                timer=timer,
            )
            data.append((case.label, result_dict))
        output_paths.extend(dump_bench_result(dataset, data, result_dir))
    return tuple(output_paths)


def bench_attention(
    attn_type: str,
    Batch: int,
    head: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim_qk: int,
    dim_v: int,
    head_k: int | None = None,
    head_v: int | None = None,
    dtype: torch.dtype = torch.float16,
    require_grad: bool = True,
    chunk_size: int | None = None,
    block_size: int | None = None,
    sparse_ratio: float | None = None,
    timer: Callable[..., float] | None = None,
):
    """Dispatch one Figure 14 case without importing optional baselines."""
    if head_k is None:
        head_k = head
    if head_v is None:
        head_v = head

    if attn_type == "causal_softmax_attn":
        return bench_softmaxattention(
            Batch,
            head,
            seqlen_q,
            seqlen_kv,
            dim_qk,
            dim_v,
            dtype=dtype,
            require_grad=require_grad,
            timer=timer,
        )
    if attn_type == "relu_attn":
        return bench_reluattention(
            Batch,
            head,
            seqlen_q,
            dim_qk,
            dim_v,
            dtype=dtype,
            require_grad=require_grad,
            timer=timer,
        )
    if attn_type == "retention_recurrent":
        return bench_retnet_recurrent(
            Batch,
            head,
            seqlen_q,
            dim_qk,
            dim_v,
            dtype=dtype,
            require_grad=require_grad,
            timer=timer,
        )
    if attn_type == "mamba2_ssm":
        return bench_mamba2_ssm(
            Batch,
            head,
            seqlen_q,
            dim_qk,
            dim_v,
            HK=head_k,
            HV=head_v,
            dtype=dtype,
            require_grad=require_grad,
            chunk_size=chunk_size,
            timer=timer,
        )
    if attn_type == "mla_attn":
        return bench_mla_decode(
            Batch,
            head,
            seqlen_kv,
            dim_qk,
            dim_v,
            HKV=head_k,
            dtype=dtype,
            timer=timer,
        )
    print(f"Warning: Undefined attention type {attn_type}, skipping benchmark.")
    return {}


def bench_softmaxattention(
    B,
    H,
    Sq,
    S,
    D,
    DV,
    device="cuda",
    dtype=torch.float16,
    require_grad=True,
    timer: Callable[..., float] | None = None,
):
    timer = _resolve_timer(timer)
    query = torch.randn(
        B, Sq, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad
    )
    do = torch.randn(B, Sq, H, DV, device=device, dtype=dtype, requires_grad=False)

    if Sq < S:
        assert not require_grad
        attention_module = softmax_attention_decode(B, H, Sq, S, D, DV)
    else:
        attention_module = causal_softmax_attention(B, H, S, D, DV, tune=True)

    def ours():
        return attention_module(query, key, value)

    ours_fwd_lat = timer(ours)
    if require_grad:
        output = attention_module(query, key, value)
        ours_bwd_lat = timer(lambda: output.backward(do, retain_graph=True))
    else:
        ours_bwd_lat = None

    result_dict = {"MetaAttention": (ours_fwd_lat, ours_bwd_lat)}

    try:
        from flash_attn import flash_attn_func

        def fa2(dim_padded):
            query_padded = (
                F.pad(query, (0, dim_padded - D), value=0.0)
                if D < dim_padded
                else query
            )
            key_padded = (
                F.pad(key, (0, dim_padded - D), value=0.0) if D < dim_padded else key
            )
            value_padded = (
                F.pad(value, (0, dim_padded - DV), value=0.0)
                if DV < dim_padded
                else value
            )
            output = flash_attn_func(
                query_padded,
                key_padded,
                value_padded,
                softmax_scale=(1 / D) ** 0.5,
                causal=True,
            )
            return output[:, :, :, :DV] if DV < dim_padded else output

        dim_padded = max(D, DV)
        fa2_fwd_lat = timer(lambda: fa2(dim_padded))
        if require_grad:
            output = fa2(dim_padded)
            fa2_bwd_lat = timer(lambda: output.backward(do, retain_graph=True))
        else:
            fa2_bwd_lat = None
        result_dict["FlashAttention-2"] = (fa2_fwd_lat, fa2_bwd_lat)
    except Exception as exc:
        print(f"Warning: FlashAttention-2 not available: {exc}")

    return result_dict


def bench_reluattention(
    B,
    H,
    S,
    D,
    DV,
    device="cuda",
    dtype=torch.float16,
    require_grad=True,
    timer: Callable[..., float] | None = None,
):
    timer = _resolve_timer(timer)
    query = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    key = torch.randn(
        B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad
    )
    value = torch.randn(
        B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad
    )
    do = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=False)

    attention_module = relu_attention(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = timer(lambda: attention_module(query, key, value))
    if require_grad:
        output = attention_module(query, key, value)
        bwd_lat = timer(lambda: output.backward(do, retain_graph=True))
    else:
        bwd_lat = None
    result_dict = {"MetaAttention": (fwd_lat, bwd_lat)}

    def ref_program(query, key, value):
        qk = torch.einsum("bqhd,bkhd->bhqk", query, key)
        qk = qk / (D**0.5)
        qk = F.relu(qk)
        return torch.einsum("bhqk,bkhd->bqhd", qk, value)

    ref_fwd_lat = timer(lambda: ref_program(query, key, value))
    if require_grad:
        output = ref_program(query, key, value)
        ref_bwd_lat = timer(lambda: output.backward(do, retain_graph=True))
    else:
        ref_bwd_lat = None
    result_dict["Torch Inductor"] = (ref_fwd_lat, ref_bwd_lat)
    return result_dict


def bench_retnet_recurrent(
    B,
    H,
    S,
    D,
    DV,
    device="cuda",
    dtype=torch.bfloat16,
    require_grad=True,
    timer: Callable[..., float] | None = None,
):
    timer = _resolve_timer(timer)
    accum_dtype = torch.float32
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    g = torch.arange(H, dtype=accum_dtype, device=device)
    g = (1 - torch.exp2(-5 - g))[None, :, None].expand(B, H, S).contiguous()
    v = torch.randn(B, H, S, DV, device=device, dtype=dtype)
    do = torch.randn(B, H, S, DV, device=device, dtype=dtype)

    q.requires_grad_(require_grad)
    k.requires_grad_(require_grad)
    v.requires_grad_(require_grad)
    q1, k1, v1 = q.clone(), k.clone(), v.clone()
    q1.requires_grad_(require_grad)
    k1.requires_grad_(require_grad)
    v1.requires_grad_(require_grad)

    attention_module = retnet_recurrent(B, H, S, D, DV, dtype=dtype, tune=True)
    fwd_lat = timer(lambda: attention_module(q, k, v, g))
    if require_grad:
        output = attention_module(q, k, v, g)
        bwd_lat = timer(lambda: output.backward(do, retain_graph=True))
    else:
        bwd_lat = None
    result_dict = {"MetaAttention": (fwd_lat, bwd_lat)}

    try:
        from fla.ops.retention import chunk_retention

        ref_fwd_lat = timer(lambda: chunk_retention(q1, k1, v1, head_first=True)[0])
        if require_grad:
            output, _ = chunk_retention(q1, k1, v1, head_first=True)
            ref_bwd_lat = timer(lambda: output.backward(do, retain_graph=True))
        else:
            ref_bwd_lat = None
        result_dict["FlashLinearAttention"] = (ref_fwd_lat, ref_bwd_lat)
    except Exception as exc:
        print(f"Warning: fla.ops.retention not available: {exc}")

    return result_dict


def bench_mamba2_ssm(
    B,
    HQ,
    S,
    D,
    DV,
    HK=None,
    HV=None,
    device="cuda",
    dtype=torch.bfloat16,
    require_grad=True,
    chunk_size: int | None = None,
    timer: Callable[..., float] | None = None,
):
    timer = _resolve_timer(timer)
    HK = HQ if HK is None else HK
    HV = HQ if HV is None else HV
    chunk_size = 32 if chunk_size is None else chunk_size

    query = torch.randn(B, S, HQ, D, device=device, dtype=dtype)
    key = torch.randn(B, S, HK, D, device=device, dtype=dtype)
    value = torch.randn(B, S, HV, DV, device=device, dtype=dtype)
    do = (
        0.1 * torch.randn(B, S, HV, DV, dtype=dtype, device=device)
        if require_grad
        else None
    )
    A_mamba = 1.5 * torch.randn(HV, dtype=dtype, device=device) - 4.0
    accum_dtype = torch.float32
    dt_mamba = 0.7 * torch.randn(B, S, HV, dtype=accum_dtype, device=device)
    dt_min, dt_max = 0.001, 0.1
    dt = torch.exp(
        torch.rand(HV, device=device, dtype=dtype)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp_min(1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)

    q_ours = query.transpose(1, 2).contiguous().detach().requires_grad_(require_grad)
    k_ours = key.transpose(1, 2).contiguous().detach().requires_grad_(require_grad)
    v_ours = value.transpose(1, 2).contiguous().detach().requires_grad_(require_grad)
    A_ours = A_mamba[None, :].clone().detach().requires_grad_(require_grad)
    dt_ours = (
        dt_mamba.transpose(1, 2).contiguous().detach().requires_grad_(require_grad)
    )
    do_ours = do.transpose(1, 2).contiguous() if do is not None else None

    attention_module = mamba2(B, HQ, S, D, DV, HK, HV, dtype=dtype, tune=True)
    fwd_lat = timer(
        lambda: attention_module(
            q_ours, k_ours, v_ours, dt_ours, A_ours, dt_ours.to(dtype)
        )
    )
    if require_grad:
        output = attention_module(
            q_ours, k_ours, v_ours, dt_ours, A_ours, dt_ours.to(dtype)
        )
        bwd_lat = timer(lambda: output.backward(do_ours, retain_graph=True))
    else:
        bwd_lat = None
    result_dict = {"MetaAttention": (fwd_lat, bwd_lat)}

    query.requires_grad_(require_grad)
    key.requires_grad_(require_grad)
    value.requires_grad_(require_grad)
    A_mamba.requires_grad_(require_grad)
    dt_mamba.requires_grad_(require_grad)
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        ref_fwd_lat = timer(
            lambda: mamba_chunk_scan_combined(
                value, dt_mamba, A_mamba, key, query, chunk_size=chunk_size
            )
        )
        if require_grad:
            output = mamba_chunk_scan_combined(
                value, dt_mamba, A_mamba, key, query, chunk_size=chunk_size
            )
            ref_bwd_lat = timer(lambda: output.backward(do, retain_graph=True))
        else:
            ref_bwd_lat = None
        result_dict["Mamba2"] = (ref_fwd_lat, ref_bwd_lat)
    except Exception as exc:
        print(f"Warning: mamba2 ssm not available: {exc}")

    return result_dict


def bench_mla_decode(
    B,
    HQ,
    SKV,
    D,
    DV,
    HKV=1,
    dtype=torch.float16,
    timer: Callable[..., float] | None = None,
):
    timer = _resolve_timer(timer)
    q = torch.randn(B, 1, HQ, DV, dtype=dtype, device="cuda")
    q_pe = torch.randn(B, 1, HQ, D - DV, dtype=dtype, device="cuda")
    kv = torch.randn(B, SKV, HKV, DV, dtype=dtype, device="cuda")
    k_pe = torch.randn(B, SKV, HKV, D - DV, dtype=dtype, device="cuda")

    attention_module = mla_decode(B, HQ, SKV, D, DV, HK=HKV, HV=HKV, dtype=dtype)
    fwd_lat = timer(lambda: attention_module(q, q_pe, kv, k_pe))
    result_dict = {"MetaAttention": (fwd_lat, None)}

    try:
        import triton
        from ref.flash_mla_decode_triton import flash_mla_triton

        cache_seqlens = torch.full((B,), SKV, dtype=torch.int32, device="cuda")
        max_seqlen_pad = triton.cdiv(int(cache_seqlens.max().item()), 256) * 256
        block_size = 64
        block_table = torch.arange(
            B * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda"
        ).view(B, max_seqlen_pad // block_size)
        blocked_kv = kv.view(B * SKV // block_size, block_size, HKV, DV)
        blocked_k_pe = k_pe.view(B * SKV // block_size, block_size, HKV, D - DV)
        ref_fwd_lat = timer(
            lambda: flash_mla_triton(
                q,
                q_pe,
                block_table,
                blocked_kv,
                blocked_k_pe,
                max_seqlen_pad,
                block_size,
                B,
                1,
                cache_seqlens,
                HQ,
                HKV,
                D,
                DV,
                True,
                dtype,
            )
        )
        result_dict["MLA Triton"] = (ref_fwd_lat, None)
    except Exception as exc:
        print(f"Warning: MLA Triton not available: {exc}")

    return result_dict


def run_figure14(
    result_dir: str | Path = "results_mi250",
    output_path: str | Path = "figure14_mi250.pdf",
    timer: Callable[..., float] | None = None,
) -> tuple[tuple[Path, ...], Path]:
    """Run Figure 14 into explicit output paths and return created artifacts."""
    result_dir = Path(result_dir)
    output_path = Path(output_path)
    csv_paths = bench_fig12(result_dir=result_dir, timer=timer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from .plot_fig_mi250 import plot_figure14

    plot_figure14(result_dir, output_path)
    return csv_paths, output_path


if __name__ == "__main__":
    print("\n" + "#" * 60)
    cprint(
        " STARTING BENCHMARK (FIGURE 14 - MI250)",
        "green",
        attrs=["bold", "reverse"],
    )
    print("#" * 60 + "\n")
    start_time = __import__("time").time()
    _, output_path = run_figure14()
    elapsed = __import__("time").time() - start_time
    print("\n" + "#" * 60)
    cprint(
        f" BENCHMARK COMPLETED IN {elapsed:.2f} SECONDS",
        "green",
        attrs=["bold", "reverse"],
    )
    print("#" * 60 + "\n")
    log_success(f"Figure 14 plotted and saved to {output_path}")
