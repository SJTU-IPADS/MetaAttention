"""Native-layout normalized identity readout over exclusive chunk carries."""

from functools import lru_cache

import torch

__all__ = ["native_identity_normalized_readout"]


def _metadata(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


def _carry_view(tensor, batch, heads, chunks, rows, columns):
    """Insert or split metadata dimensions without materializing initial States."""
    shape = (batch, heads, chunks, rows, columns)
    if tensor.ndim == 5:
        return tensor
    if chunks == 1 and tuple(tensor.shape[:2]) == (batch, heads):
        sb, sh, sr, sc = tensor.stride()
        strides = (sb, sh, 0, sr, sc)
    else:
        sh, st, sr, sc = tensor.stride()
        strides = (heads * sh, sh, st, sr, sc)
    return torch.as_strided(tensor, shape, strides)


@lru_cache(maxsize=None)
def _compile_normalized_readout(
    sequence_length,
    heads,
    rows,
    columns,
    memory_meta,
    key_meta,
    value_meta,
    normalizer_meta,
    den_key_meta,
    query_meta,
    output_meta,
    prior_memory,
    prior_normalizer,
    shared_key,
    epsilon,
    query_scale,
    device_index,
):
    import tilelang
    import tilelang.language as T

    from ..diagonal.identity import fp32_gemm_macro

    gemm = fp32_gemm_macro(T)
    batch = query_meta[0][0]
    chunks = (sequence_length + 63) // 64
    key_group = heads // key_meta[0][1]
    value_group = heads // value_meta[0][1]
    den_key_group = heads // den_key_meta[0][1]
    query_group = heads // query_meta[0][1]
    query_dtype = query_meta[2]
    # TVM folds nested constant casts through BF16; round the scalar first.
    rounded_scale = (
        None
        if query_scale is None
        else torch.tensor(query_scale, dtype=getattr(torch, query_dtype)).item()
    )

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            memory: T.StridedTensor(memory_meta[0], memory_meta[1], memory_meta[2]),
            key: T.StridedTensor(key_meta[0], key_meta[1], key_meta[2]),
            value: T.StridedTensor(value_meta[0], value_meta[1], value_meta[2]),
            normalizer: T.StridedTensor(
                normalizer_meta[0], normalizer_meta[1], normalizer_meta[2]
            ),
            den_key: T.StridedTensor(den_key_meta[0], den_key_meta[1], den_key_meta[2]),
            query: T.StridedTensor(query_meta[0], query_meta[1], query_meta[2]),
            output: T.StridedTensor(output_meta[0], output_meta[1], output_meta[2]),
        ):
            with T.Kernel(
                batch * heads, chunks, T.ceildiv(columns, 32), threads=128
            ) as (batch_head, chunk, column_block):
                sequence = batch_head // heads
                head = batch_head % heads
                query_shared = T.alloc_shared((64, 32), "float32")
                key_shared = T.alloc_shared((32, 64), "float32")
                memory_shared = T.alloc_shared((32, 32), "float32")
                scores_shared = T.alloc_shared((64, 64), "float32")
                value_shared = T.alloc_shared((64, 32), "float32")
                scores = T.alloc_fragment((64, 64), "float32")
                den_scores = T.alloc_fragment((64, 64), "float32")
                numerator = T.alloc_fragment((64, 32), "float32")
                denominator = T.alloc_shared((64,), "float32")
                den_products = T.alloc_fragment((64, 32), "float32")
                den_block = T.alloc_fragment((64,), "float32")
                den_score_sum = T.alloc_fragment((64,), "float32")
                T.clear(scores)
                T.clear(numerator)
                T.clear(denominator)
                if not shared_key:
                    T.clear(den_scores)

                for row_block in T.serial(T.ceildiv(rows, 32)):
                    for token, row in T.Parallel(64, 32):
                        if (chunk * 64 + token < sequence_length) & (
                            row_block * 32 + row < rows
                        ):
                            if query_scale is not None:
                                # Authored scalar conversion and multiply round to
                                # the query dtype before the FP32 contraction seam.
                                query_shared[token, row] = T.cast(
                                    T.cast(
                                        T.cast(
                                            query[
                                                sequence,
                                                head // query_group,
                                                chunk * 64 + token,
                                                row_block * 32 + row,
                                            ],
                                            "float32",
                                        )
                                        * rounded_scale,
                                        query_dtype,
                                    ),
                                    "float32",
                                )
                            else:
                                query_shared[token, row] = T.cast(
                                    query[
                                        sequence,
                                        head // query_group,
                                        chunk * 64 + token,
                                        row_block * 32 + row,
                                    ],
                                    "float32",
                                )
                        else:
                            query_shared[token, row] = 0.0
                    for row, token in T.Parallel(32, 64):
                        key_shared[row, token] = T.if_then_else(
                            (row_block * 32 + row < rows)
                            & (chunk * 64 + token < sequence_length),
                            T.cast(
                                key[
                                    sequence,
                                    head // key_group,
                                    chunk * 64 + token,
                                    row_block * 32 + row,
                                ],
                                "float32",
                            ),
                            0.0,
                        )
                    for row, column in T.Parallel(32, 32):
                        memory_shared[row, column] = T.if_then_else(
                            (row_block * 32 + row < rows)
                            & (column_block * 32 + column < columns),
                            T.cast(
                                memory[
                                    sequence,
                                    head,
                                    chunk,
                                    row_block * 32 + row,
                                    column_block * 32 + column,
                                ],
                                "float32",
                            ),
                            0.0,
                        )
                    gemm(query_shared, key_shared, scores)
                    gemm(query_shared, memory_shared, numerator)
                    for token, row in T.Parallel(64, 32):
                        den_products[token, row] = query_shared[
                            token, row
                        ] * T.if_then_else(
                            row_block * 32 + row < rows,
                            T.cast(
                                normalizer[
                                    sequence, head, chunk, row_block * 32 + row, 0
                                ],
                                "float32",
                            ),
                            0.0,
                        )
                    T.reduce_sum(den_products, den_block, dim=1)
                    for token in T.Parallel(64):
                        denominator[token] += den_block[token]
                    if not shared_key:
                        for row, token in T.Parallel(32, 64):
                            key_shared[row, token] = T.if_then_else(
                                (row_block * 32 + row < rows)
                                & (chunk * 64 + token < sequence_length),
                                T.cast(
                                    den_key[
                                        sequence,
                                        head // den_key_group,
                                        chunk * 64 + token,
                                        row_block * 32 + row,
                                    ],
                                    "float32",
                                ),
                                0.0,
                            )
                        gemm(query_shared, key_shared, den_scores)

                for token, source in T.Parallel(64, 64):
                    if shared_key:
                        den_scores[token, source] = scores[token, source]
                    den_scores[token, source] = T.if_then_else(
                        (chunk * 64 + token < sequence_length)
                        & (chunk * 64 + source < sequence_length)
                        & ((source < token) if prior_normalizer else (source <= token)),
                        den_scores[token, source],
                        0.0,
                    )
                    scores[token, source] = T.if_then_else(
                        (chunk * 64 + token < sequence_length)
                        & (chunk * 64 + source < sequence_length)
                        & ((source < token) if prior_memory else (source <= token)),
                        scores[token, source],
                        0.0,
                    )
                T.reduce_sum(den_scores, den_score_sum, dim=1)
                for token in T.Parallel(64):
                    denominator[token] += den_score_sum[token]
                T.copy(scores, scores_shared)
                for source, column in T.Parallel(64, 32):
                    value_shared[source, column] = T.if_then_else(
                        (chunk * 64 + source < sequence_length)
                        & (column_block * 32 + column < columns),
                        T.cast(
                            value[
                                sequence,
                                head // value_group,
                                chunk * 64 + source,
                                column_block * 32 + column,
                            ],
                            "float32",
                        ),
                        0.0,
                    )
                gemm(scores_shared, value_shared, numerator)
                for token, column in T.Parallel(64, 32):
                    if (chunk * 64 + token < sequence_length) & (
                        column_block * 32 + column < columns
                    ):
                        output[
                            sequence * sequence_length + chunk * 64 + token,
                            head,
                            column_block * 32 + column,
                        ] = numerator[token, column] / (
                            denominator[token] + T.cast(epsilon, "float32")
                        )

        return kernel

    return build_kernel()


def native_identity_normalized_readout(
    memory,
    key,
    value,
    normalizer,
    den_key,
    query,
    output,
    *,
    sequence_length,
    heads,
    prior_memory,
    prior_normalizer,
    shared_key,
    epsilon,
    query_scale,
):
    """Write normalized outputs from native vector views without mutating carries."""
    batch = query.shape[0]
    rows, columns = memory.shape[-2:]
    chunks = (sequence_length + 63) // 64
    memory = _carry_view(memory, batch, heads, chunks, rows, columns)
    normalizer = _carry_view(normalizer, batch, heads, chunks, rows, 1)
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _compile_normalized_readout(
        sequence_length,
        heads,
        rows,
        columns,
        _metadata(memory),
        _metadata(key),
        _metadata(value),
        _metadata(normalizer),
        _metadata(den_key),
        _metadata(query),
        _metadata(output),
        prior_memory,
        prior_normalizer,
        shared_key,
        epsilon,
        query_scale,
        device_index,
    )(memory, key, value, normalizer, den_key, query, output)
    return output
