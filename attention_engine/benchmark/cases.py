from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One measured point in a paper figure's benchmark matrix."""

    case_id: str
    dataset: str
    attn_type: str
    batch: int
    heads: int
    seqlen_q: int
    seqlen_kv: int
    dim_qk: int
    dim_v: int
    head_k: int | None = None
    head_v: int | None = None
    dtype: str = "float16"
    require_grad: bool = True
    chunk_size: int | None = None
    block_size: int | None = None
    sparse_ratio: float | None = None

    @property
    def label(self) -> str:
        if self.seqlen_q == 1 and self.seqlen_q != self.seqlen_kv:
            return f"BS{self.batch}S1\nKV{self.seqlen_kv}"
        return f"BS{self.batch}\nS{self.seqlen_kv}"

    # Short aliases make the matrix convenient to inspect in notebooks and
    # retain the terminology used in the benchmark scripts and paper tables.
    @property
    def B(self) -> int:
        return self.batch

    @property
    def H(self) -> int:
        return self.heads

    @property
    def Sq(self) -> int:
        return self.seqlen_q

    @property
    def S(self) -> int:
        return self.seqlen_kv

    @property
    def D(self) -> int:
        return self.dim_qk

    @property
    def DV(self) -> int:
        return self.dim_v

    @property
    def HKV(self) -> int:
        return self.head_k if self.head_k is not None else self.heads

    @property
    def dtype_name(self) -> str:
        return self.dtype


def _case(
    dataset: str,
    attn_type: str,
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim_qk: int,
    dim_v: int,
    *,
    case_number: int,
    head_k: int | None = None,
    head_v: int | None = None,
    dtype: str = "float16",
    require_grad: bool = True,
    chunk_size: int | None = None,
    block_size: int | None = None,
    sparse_ratio: float | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=f"{dataset}-{case_number:02d}",
        dataset=dataset,
        attn_type=attn_type,
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        dim_qk=dim_qk,
        dim_v=dim_v,
        head_k=head_k,
        head_v=head_v,
        dtype=dtype,
        require_grad=require_grad,
        chunk_size=chunk_size,
        block_size=block_size,
        sparse_ratio=sparse_ratio,
    )


def _product_cases(
    dataset: str,
    attn_type: str,
    batches: tuple[int, ...],
    seqlens: tuple[int, ...],
    heads: int,
    dim_qk: int,
    dim_v: int,
    *,
    start: int = 1,
    seqlen_q: int | None = None,
    head_k: int | None = None,
    head_v: int | None = None,
    dtype: str = "float16",
    require_grad: bool = True,
    chunk_size: int | None = None,
    block_size: int | None = None,
    sparse_ratio: float | None = None,
) -> tuple[BenchmarkCase, ...]:
    return tuple(
        _case(
            dataset,
            attn_type,
            batch,
            heads,
            seqlen if seqlen_q is None else seqlen_q,
            seqlen,
            dim_qk,
            dim_v,
            case_number=start + index,
            head_k=head_k,
            head_v=head_v,
            dtype=dtype,
            require_grad=require_grad,
            chunk_size=chunk_size,
            block_size=block_size,
            sparse_ratio=sparse_ratio,
        )
        for index, (batch, seqlen) in enumerate(product(batches, seqlens))
    )


def h100_cases() -> tuple[BenchmarkCase, ...]:
    """Return the exact 62-point Figure 11 matrix in display order."""

    cases: list[BenchmarkCase] = []
    batches = (1, 8)
    long_seqlens = (2048, 4096, 8192)

    cases.extend(
        _product_cases(
            "deepseek",
            "causal_softmax_attn",
            batches,
            long_seqlens,
            16,
            192,
            128,
            start=1,
        )
    )
    cases.extend(
        _product_cases(
            "deepseek",
            "causal_softmax_attn",
            batches,
            long_seqlens,
            16,
            192,
            128,
            start=7,
            seqlen_q=1,
            require_grad=False,
        )
    )
    cases.extend(
        _product_cases(
            "llama", "causal_softmax_attn", batches, long_seqlens, 32, 128, 128
        )
    )
    cases.extend(
        _product_cases(
            "vit", "relu_attn", (32, 64), (512, 1024, 2048), 6, 64, 64, start=1
        )
    )
    cases.extend(
        _product_cases(
            "dit", "causal_softmax_attn", batches, long_seqlens, 12, 128, 256
        )
    )
    cases.extend(
        _product_cases(
            "retnet", "retention_parallel", batches, (2048, 4096), 32, 256, 512
        )
    )
    cases.extend(
        _product_cases(
            "sigmoid_attn", "sigmoid_attn", batches, long_seqlens, 32, 128, 128
        )
    )
    cases.extend(
        _product_cases(
            "rfa",
            "gated_retention",
            (64,),
            (1024, 2048, 4096),
            16,
            64,
            64,
            dtype="bfloat16",
        )
    )
    cases.extend(
        _product_cases(
            "mamba2",
            "mamba2_ssm",
            batches,
            long_seqlens,
            1,
            128,
            64,
            head_k=1,
            head_v=80,
            dtype="bfloat16",
            chunk_size=64,
        )
    )
    cases.extend(
        _product_cases(
            "yoco",
            "gated_retention",
            (8,),
            (1024, 2048, 4096),
            40,
            256,
            256,
            dtype="bfloat16",
        )
    )
    cases.extend(
        _product_cases(
            "retnet_recur",
            "retention_recurrent",
            batches,
            (2048, 4096),
            32,
            256,
            512,
            dtype="bfloat16",
        )
    )
    cases.extend(
        _product_cases(
            "mla",
            "mla_attn",
            (8,),
            long_seqlens,
            128,
            576,
            512,
            seqlen_q=1,
            head_k=1,
            head_v=1,
            dtype="bfloat16",
        )
    )
    cases.extend(
        _product_cases(
            "sparse_gqa",
            "sparse_gqa",
            (8,),
            long_seqlens,
            32,
            128,
            128,
            seqlen_q=1,
            head_k=8,
            head_v=8,
            block_size=32,
            sparse_ratio=0.8,
        )
    )
    return tuple(cases)


def mi250_cases() -> tuple[BenchmarkCase, ...]:
    """Return the exact 25-point Figure 14 matrix in display order."""

    batches = (1, 8)
    long_seqlens = (2048, 4096, 8192)
    cases: list[BenchmarkCase] = []
    cases.extend(
        _product_cases(
            "deepseek", "causal_softmax_attn", batches, long_seqlens, 16, 192, 128
        )
    )
    cases.extend(
        _product_cases("vit", "relu_attn", (32, 64), (512, 1024, 2048), 6, 64, 64)
    )
    cases.extend(
        _product_cases(
            "mamba2",
            "mamba2_ssm",
            batches,
            long_seqlens,
            1,
            128,
            64,
            head_k=1,
            head_v=80,
            dtype="bfloat16",
            chunk_size=32,
        )
    )
    cases.extend(
        _product_cases(
            "retnet_recur",
            "retention_recurrent",
            batches,
            (2048, 4096),
            32,
            256,
            512,
            dtype="bfloat16",
        )
    )
    cases.extend(
        _product_cases(
            "mla",
            "mla_attn",
            (8,),
            long_seqlens,
            128,
            576,
            512,
            seqlen_q=1,
            head_k=1,
            head_v=1,
            dtype="float16",
        )
    )
    return tuple(cases)


__all__ = ["BenchmarkCase", "h100_cases", "mi250_cases"]
