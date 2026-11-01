"""TileLang identity summaries and chunk contractions, with FP32 carries."""

from functools import lru_cache

import torch


def fp32_gemm_macro(T):
    """Three TensorFloat products; retain residuals instead of single-TF32 rounding."""

    @T.macro
    def gemm(A, B, C):
        ah = T.alloc_shared(A.shape, T.tfloat32)
        al = T.alloc_shared(A.shape, T.tfloat32)
        bh = T.alloc_shared(B.shape, T.tfloat32)
        bl = T.alloc_shared(B.shape, T.tfloat32)
        for i, k in T.Parallel(A.shape[0], A.shape[1]):
            ah[i, k] = T.cast(A[i, k], T.tfloat32)
            al[i, k] = T.cast(A[i, k] - T.cast(ah[i, k], "float32"), T.tfloat32)
        for k, j in T.Parallel(B.shape[0], B.shape[1]):
            bh[k, j] = T.cast(B[k, j], T.tfloat32)
            bl[k, j] = T.cast(B[k, j] - T.cast(bh[k, j], "float32"), T.tfloat32)
        T.gemm(al, bh, C)
        T.gemm(ah, bl, C)
        T.gemm(ah, bh, C)

    return gemm


def _metadata(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


def native_vector_view(tensor, batch, heads, length, features, strides, group):
    """Canonical logical axes over native storage, including grouped heads."""
    return torch.as_strided(tensor, (batch, heads // group, length, features), strides)


def _state_view(tensor, batch, heads, chunks):
    rows, columns = tensor.shape[-2:]
    if tensor.shape[:2] == (batch, heads) and chunks == 1:
        sb, sh, sr, sc = tensor.stride()
        strides = (sb, sh, 0, sr, sc)
    else:
        sh, st, sr, sc = tensor.stride()
        strides = (heads * sh, sh, st, sr, sc)
    return torch.as_strided(tensor, (batch, heads, chunks, rows, columns), strides)


def _summary_macro(T, heads, length, kr, unit_value, add_initial):
    gemm = fp32_gemm_macro(T)

    @T.macro
    def summary(Key, Key2, Value, Bias, Initial, sh, chunk, tile):
        rows, columns = Bias.shape[-2], Bias.shape[-1]
        kg, vg = heads // Key.shape[1], heads // Value.shape[1]
        seq, head = sh // heads, sh % heads
        rb, cb = tile // T.ceildiv(columns, 32), tile % T.ceildiv(columns, 32)
        k = T.alloc_shared((32, 64), "float32")
        v = T.alloc_shared((64, 32), "float32")
        result = T.alloc_fragment((32, 32), "float32")
        for r, t in T.Parallel(32, 64):
            if (rb * 32 + r < rows) & (chunk * 64 + t < length):
                if kr:
                    k[r, t] = T.cast(
                        Key[seq, head // kg, chunk * 64 + t, (rb * 32 + r) // kr],
                        "float32",
                    ) * T.cast(
                        Key2[seq, head // kg, chunk * 64 + t, (rb * 32 + r) % kr],
                        "float32",
                    )
                else:
                    k[r, t] = T.cast(
                        Key[seq, head // kg, chunk * 64 + t, rb * 32 + r], "float32"
                    )
            else:
                k[r, t] = 0.0
        for t, c in T.Parallel(64, 32):
            if (chunk * 64 + t < length) & (cb * 32 + c < columns):
                if unit_value:
                    v[t, c] = 1.0
                else:
                    v[t, c] = T.cast(
                        Value[seq, head // vg, chunk * 64 + t, cb * 32 + c], "float32"
                    )
            else:
                v[t, c] = 0.0
        T.clear(result)
        gemm(k, v, result)
        for r, c in T.Parallel(32, 32):
            if (rb * 32 + r < rows) & (cb * 32 + c < columns):
                if add_initial:
                    result[r, c] += Initial[seq, head, 0, rb * 32 + r, cb * 32 + c]
                Bias[seq, head, chunk, rb * 32 + r, cb * 32 + c] = result[r, c]

    return summary


@lru_cache(maxsize=None)
def _compile_summary(
    key_meta,
    key2_meta,
    value_meta,
    bias_meta,
    initial_meta,
    heads,
    length,
    kr,
    unit_value,
    add_initial,
    device_index,
):
    import tilelang
    import tilelang.language as T

    summary = _summary_macro(T, heads, length, kr, unit_value, add_initial)
    batch, _, chunks, rows, columns = bias_meta[0]

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Key: T.StridedTensor(*key_meta),
            Key2: T.StridedTensor(*key2_meta),
            Value: T.StridedTensor(*value_meta),
            Bias: T.StridedTensor(*bias_meta),
            Initial: T.StridedTensor(*initial_meta),
        ):
            with T.Kernel(
                batch * heads,
                chunks,
                T.ceildiv(rows, 32) * T.ceildiv(columns, 32),
                threads=128,
            ) as (sh, chunk, tile):
                summary(Key, Key2, Value, Bias, Initial, sh, chunk, tile)

        return kernel

    return build_kernel()


@lru_cache(maxsize=None)
def _compile_summary_pair(
    key_meta,
    den_key_meta,
    value_meta,
    bias_meta,
    den_bias_meta,
    initial_meta,
    den_initial_meta,
    heads,
    length,
    add_initial,
    device_index,
):
    import tilelang
    import tilelang.language as T

    summary = _summary_macro(T, heads, length, 0, False, add_initial)
    den_summary = _summary_macro(T, heads, length, 0, True, add_initial)
    batch, _, chunks, rows, columns = bias_meta[0]
    tiles = ((rows + 31) // 32) * ((columns + 31) // 32)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Key: T.StridedTensor(*key_meta),
            DenKey: T.StridedTensor(*den_key_meta),
            Value: T.StridedTensor(*value_meta),
            Bias: T.StridedTensor(*bias_meta),
            DenBias: T.StridedTensor(*den_bias_meta),
            Initial: T.StridedTensor(*initial_meta),
            DenInitial: T.StridedTensor(*den_initial_meta),
        ):
            with T.Kernel(
                batch * heads, chunks, tiles + T.ceildiv(rows, 32), threads=128
            ) as (sh, chunk, tile):
                if tile < tiles:
                    summary(Key, Key, Value, Bias, Initial, sh, chunk, tile)
                else:
                    den_summary(
                        DenKey,
                        DenKey,
                        Value,
                        DenBias,
                        DenInitial,
                        sh,
                        chunk,
                        tile - tiles,
                    )

        return kernel

    return build_kernel()


def identity_summary_pair(
    key,
    den_key,
    value,
    bias,
    den_bias,
    *,
    sequence_length,
    heads,
    initial=None,
    den_initial=None,
):
    batch = key.shape[0]
    chunks = 1 if initial is not None else (sequence_length + 63) // 64
    output = _state_view(bias, batch, heads, chunks)
    den_output = _state_view(den_bias, batch, heads, chunks)
    initial_view = output if initial is None else _state_view(initial, batch, heads, 1)
    den_initial_view = (
        den_output if den_initial is None else _state_view(den_initial, batch, heads, 1)
    )
    _compile_summary_pair(
        *(
            _metadata(t)
            for t in (
                key,
                den_key,
                value,
                output,
                den_output,
                initial_view,
                den_initial_view,
            )
        ),
        heads,
        sequence_length,
        initial is not None,
        key.device.index,
    )(key, den_key, value, output, den_output, initial_view, den_initial_view)


def identity_summary(
    key,
    key2,
    value,
    bias,
    *,
    sequence_length,
    heads,
    kr=0,
    unit_value=False,
    initial=None,
):
    batch = key.shape[0]
    chunks = 1 if initial is not None else (sequence_length + 63) // 64
    if not chunks:
        return
    output = _state_view(bias, batch, heads, chunks)
    initial_view = output if initial is None else _state_view(initial, batch, heads, 1)
    _compile_summary(
        _metadata(key),
        _metadata(key2),
        _metadata(value),
        _metadata(output),
        _metadata(initial_view),
        heads,
        sequence_length,
        kr,
        unit_value,
        initial is not None,
        key.device.index,
    )(key, key2, value, output, initial_view)


@lru_cache(maxsize=None)
def _compile_readout(
    key_meta,
    key2_meta,
    value_meta,
    query_meta,
    query2_meta,
    carry_meta,
    output_meta,
    heads,
    length,
    kr,
    qr,
    prior,
    device_index,
):
    import tilelang
    import tilelang.language as T

    gemm = fp32_gemm_macro(T)
    batch, _, chunks, rows, columns = carry_meta[0]

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Key: T.StridedTensor(*key_meta),
            Key2: T.StridedTensor(*key2_meta),
            Value: T.StridedTensor(*value_meta),
            Query: T.StridedTensor(*query_meta),
            Query2: T.StridedTensor(*query2_meta),
            Carry: T.StridedTensor(*carry_meta),
            Output: T.StridedTensor(*output_meta),
        ):
            with T.Kernel(
                batch * heads, chunks, T.ceildiv(columns, 32), threads=128
            ) as (sh, chunk, cb):
                seq, head = sh // heads, sh % heads
                q = T.alloc_shared((64, 32), "float32")
                k = T.alloc_shared((32, 64), "float32")
                state = T.alloc_shared((32, 32), "float32")
                v = T.alloc_shared((64, 32), "float32")
                score_shared = T.alloc_shared((64, 64), "float32")
                scores = T.alloc_fragment((64, 64), "float32")
                output = T.alloc_fragment((64, 32), "float32")
                T.clear(scores)
                T.clear(output)
                for rb in T.serial(T.ceildiv(rows, 32)):
                    for t, r in T.Parallel(64, 32):
                        if (chunk * 64 + t < length) & (rb * 32 + r < rows):
                            if qr:
                                q[t, r] = (
                                    Query[
                                        seq, head, chunk * 64 + t, (rb * 32 + r) // qr
                                    ]
                                    * Query2[
                                        seq, head, chunk * 64 + t, (rb * 32 + r) % qr
                                    ]
                                )
                            else:
                                q[t, r] = Query[seq, head, chunk * 64 + t, rb * 32 + r]
                        else:
                            q[t, r] = 0.0
                    for r, t in T.Parallel(32, 64):
                        if (chunk * 64 + t < length) & (rb * 32 + r < rows):
                            if kr:
                                k[r, t] = (
                                    Key[seq, head, chunk * 64 + t, (rb * 32 + r) // kr]
                                    * Key2[
                                        seq, head, chunk * 64 + t, (rb * 32 + r) % kr
                                    ]
                                )
                            else:
                                k[r, t] = Key[seq, head, chunk * 64 + t, rb * 32 + r]
                        else:
                            k[r, t] = 0.0
                    for r, c in T.Parallel(32, 32):
                        state[r, c] = T.if_then_else(
                            (rb * 32 + r < rows) & (cb * 32 + c < columns),
                            Carry[seq, head, chunk, rb * 32 + r, cb * 32 + c],
                            0.0,
                        )
                    gemm(q, k, scores)
                    gemm(q, state, output)
                for t, s in T.Parallel(64, 64):
                    score_shared[t, s] = T.if_then_else(
                        (s < t) if prior else (s <= t), scores[t, s], 0.0
                    )
                for s, c in T.Parallel(64, 32):
                    v[s, c] = T.if_then_else(
                        (chunk * 64 + s < length) & (cb * 32 + c < columns),
                        Value[seq, head, chunk * 64 + s, cb * 32 + c],
                        0.0,
                    )
                gemm(score_shared, v, output)
                for t, c in T.Parallel(64, 32):
                    if (chunk * 64 + t < length) & (cb * 32 + c < columns):
                        Output[seq * length + chunk * 64 + t, head, cb * 32 + c] = (
                            output[t, c]
                        )

        return kernel

    return build_kernel()


def identity_readout(
    key,
    key2,
    value,
    query,
    query2,
    carry,
    output,
    *,
    sequence_length,
    heads,
    kr=0,
    qr=0,
    prior=False,
):
    batch = key.shape[0]
    canonical_carry = _state_view(carry, batch, heads, (sequence_length + 63) // 64)
    _compile_readout(
        *(
            _metadata(t)
            for t in (key, key2, value, query, query2, canonical_carry, output)
        ),
        heads,
        sequence_length,
        kr,
        qr,
        prior,
        key.device.index,
    )(key, key2, value, query, query2, canonical_carry, output)
