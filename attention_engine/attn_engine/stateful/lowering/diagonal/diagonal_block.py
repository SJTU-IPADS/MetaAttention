"""Inverse-free TileLang diagonal summaries and chunk-local readouts.

Each summary lane owns one State element. Each readout lane owns a State
column and advances rows in authored order; no token-State scratch, inverse
products, or materialized ProductFeature vectors are required.
"""

from functools import lru_cache

import torch


def _metadata(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


@lru_cache(maxsize=None)
def _compile_summary(
    key_meta,
    key2_meta,
    value_meta,
    row_meta,
    column_meta,
    scale_meta,
    bias_meta,
    length,
    heads,
    kr,
    unit_rows,
    unit_columns,
    device_index,
):
    import tilelang
    import tilelang.language as T

    bh, chunks, rows, columns = bias_meta[0]

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Key: T.StridedTensor(*key_meta),
            Key2: T.StridedTensor(*key2_meta),
            Value: T.StridedTensor(*value_meta),
            Row: T.StridedTensor(*row_meta),
            Column: T.StridedTensor(*column_meta),
            Scale: T.StridedTensor(*scale_meta),
            Bias: T.StridedTensor(*bias_meta),
        ):
            with T.Kernel(bh, chunks, T.ceildiv(rows * columns, 128), threads=128) as (
                sh,
                chunk,
                block,
            ):
                x = block * 128 + T.get_thread_binding()
                r, c = x // columns, x % columns
                seq, head = sh // heads, sh % heads
                row_scale = T.alloc_local((1,), "float32")
                column_scale = T.alloc_local((1,), "float32")
                bias = T.alloc_local((1,), "float32")
                row_scale[0], column_scale[0], bias[0] = 1.0, 1.0, 0.0
                if x < rows * columns:
                    for step in T.serial(64):
                        t = chunk * 64 + step
                        if t < length:
                            flat = seq * length + t
                            rg = (
                                1.0
                                if unit_rows
                                else T.cast(Row[flat, head, r], "float32")
                            )
                            cg = (
                                1.0
                                if unit_columns
                                else T.cast(Column[flat, head, c], "float32")
                            )
                            if kr:
                                key = T.cast(
                                    Key[flat, head, r // kr], "float32"
                                ) * T.cast(Key2[flat, head, r % kr], "float32")
                            else:
                                key = T.cast(Key[flat, head, r], "float32")
                            value = T.cast(Value[flat, head, c], "float32")
                            bias[0] = (rg * cg) * bias[0] + key * value
                            row_scale[0] *= rg
                            column_scale[0] *= cg
                    Scale[sh, chunk, r, c] = row_scale[0] * column_scale[0]
                    Bias[sh, chunk, r, c] = bias[0]

        return kernel

    return build()


def diagonal_block_summary(
    rows_gate,
    columns_gate,
    key,
    key2,
    value,
    scale,
    bias,
    *,
    sequence_length,
    heads,
    rows,
    columns,
    chunks,
    kl,
    kr,
):
    row = key if rows_gate is None else rows_gate
    column = value if columns_gate is None else columns_gate
    key2 = key if not kr else key2
    _compile_summary(
        *(_metadata(t) for t in (key, key2, value, row, column, scale, bias)),
        sequence_length,
        heads,
        kr,
        rows_gate is None,
        columns_gate is None,
        bias.device.index,
    )(key, key2, value, row, column, scale, bias)


@lru_cache(maxsize=None)
def _compile_readout(
    key_meta,
    key2_meta,
    value_meta,
    query_meta,
    query2_meta,
    row_meta,
    column_meta,
    carry_meta,
    output_meta,
    bonus_meta,
    current_meta,
    length,
    heads,
    kr,
    qr,
    unit_rows,
    unit_columns,
    prior,
    bonus_mode,
    query_scale,
    device_index,
):
    import tilelang
    import tilelang.language as T

    bh, chunks, rows, columns = carry_meta[0]
    query_dtype = query_meta[2]
    bonus_dtype = str(
        torch.promote_types(getattr(torch, query_dtype), getattr(torch, bonus_meta[2]))
    ).removeprefix("torch.")
    current_dtype = str(
        torch.promote_types(
            getattr(torch, bonus_dtype), getattr(torch, current_meta[2])
        )
    ).removeprefix("torch.")
    rounded_scale = torch.tensor(query_scale, dtype=getattr(torch, query_dtype)).item()

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build():
        @T.prim_func
        def kernel(
            Key: T.StridedTensor(*key_meta),
            Key2: T.StridedTensor(*key2_meta),
            Value: T.StridedTensor(*value_meta),
            Query: T.StridedTensor(*query_meta),
            Query2: T.StridedTensor(*query2_meta),
            Row: T.StridedTensor(*row_meta),
            Column: T.StridedTensor(*column_meta),
            Carry: T.StridedTensor(*carry_meta),
            Output: T.StridedTensor(*output_meta),
            Bonus: T.StridedTensor(*bonus_meta),
            Current: T.StridedTensor(*current_meta),
        ):
            with T.Kernel(bh, chunks, T.ceildiv(columns, 32), threads=32) as (
                sh,
                chunk,
                block,
            ):
                c = block * 32 + T.get_thread_binding()
                seq, head = sh // heads, sh % heads
                state = T.alloc_local((rows,), "float32")
                result = T.alloc_local((1,), "float32")
                bonus_score = T.alloc_local((1,), "float32")
                if c < columns:
                    for r in T.serial(rows):
                        state[r] = T.cast(Carry[sh, chunk, r, c], "float32")
                    for step in T.serial(64):
                        t = chunk * 64 + step
                        if t < length:
                            flat = seq * length + t
                            result[0], bonus_score[0] = 0.0, 0.0
                            value = T.cast(Value[flat, head, c], "float32")
                            cg = (
                                1.0
                                if unit_columns
                                else T.cast(Column[flat, head, c], "float32")
                            )
                            for r in T.serial(rows):
                                if qr:
                                    query = T.cast(
                                        Query[flat, head, r // qr], "float32"
                                    ) * T.cast(Query2[flat, head, r % qr], "float32")
                                else:
                                    query = T.cast(Query[flat, head, r], "float32")
                                if bonus_mode:
                                    query = T.cast(
                                        T.cast(query * rounded_scale, query_dtype),
                                        "float32",
                                    )
                                    weighted = T.cast(
                                        T.cast(
                                            query
                                            * T.cast(Bonus[flat, head, r], "float32"),
                                            bonus_dtype,
                                        ),
                                        "float32",
                                    )
                                    bonus_score[0] += T.cast(
                                        T.cast(
                                            weighted
                                            * T.cast(Current[flat, head, r], "float32"),
                                            current_dtype,
                                        ),
                                        "float32",
                                    )
                                if prior:
                                    result[0] += query * state[r]
                                rg = (
                                    1.0
                                    if unit_rows
                                    else T.cast(Row[flat, head, r], "float32")
                                )
                                if kr:
                                    key = T.cast(
                                        Key[flat, head, r // kr], "float32"
                                    ) * T.cast(Key2[flat, head, r % kr], "float32")
                                else:
                                    key = T.cast(Key[flat, head, r], "float32")
                                state[r] = (rg * cg) * state[r] + key * value
                                if not prior:
                                    result[0] += query * state[r]
                            if bonus_mode:
                                result[0] += bonus_score[0] * value
                            Output[flat, head, c] = result[0]

        return kernel

    return build()


def _read(
    carry,
    factors,
    query,
    query2,
    output,
    *,
    sequence_length,
    prior,
    qr,
    bonus=None,
    current_key=None,
    query_scale=1.0,
):
    row, column, key, key2, value, _, kr = factors
    unit_rows, unit_columns = row is None, column is None
    row = key if unit_rows else row
    column = value if unit_columns else column
    key2 = key if not kr else key2
    query2 = query if not qr else query2
    bonus_mode = bonus is not None
    bonus = query if bonus is None else bonus
    current_key = key if current_key is None else current_key
    _compile_readout(
        *(
            _metadata(t)
            for t in (
                key,
                key2,
                value,
                query,
                query2,
                row,
                column,
                carry,
                output,
                bonus,
                current_key,
            )
        ),
        sequence_length,
        key.shape[1],
        kr,
        qr,
        unit_rows,
        unit_columns,
        prior,
        bonus_mode,
        query_scale,
        carry.device.index,
    )(key, key2, value, query, query2, row, column, carry, output, bonus, current_key)


def diagonal_block_readout(
    carry, factors, query, query2, output, *, sequence_length, prior, ql, qr
):
    _read(
        carry,
        factors,
        query,
        query2,
        output,
        sequence_length=sequence_length,
        prior=prior,
        qr=qr,
    )


def diagonal_block_bonus_readout(
    carry, factors, query, bonus, current_key, output, *, sequence_length, query_scale
):
    """Fuse prior contraction with authored ``(query * bonus * key) @ value``."""
    _, column, key, key2, _, _, kr = factors
    if column is not None or kr or key2.data_ptr() != key.data_ptr():
        raise ValueError(
            "bonus readout requires a dense single-axis key and no column gate"
        )
    if (
        query.ndim != 3
        or bonus.shape != query.shape
        or current_key.shape != query.shape
    ):
        raise ValueError("bonus readout expects dense [tokens, heads, rows] vectors")
    _read(
        carry,
        factors,
        query,
        query,
        output,
        sequence_length=sequence_length,
        prior=True,
        qr=0,
        bonus=bonus,
        current_key=current_key,
        query_scale=query_scale,
    )
