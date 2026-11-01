"""TileLang ProductFeature readouts with chunk-local base-Feature GEMMs.

ProductFeature axes stay factored at runtime. The local score kernel computes
the product of the left and right base-feature contractions, while the
readout kernel contracts the factored query against the incoming chunk State
and adds the causal local value contribution. Both kernels use FP32
accumulation and preserve the public token-major ABI.
"""

from functools import lru_cache

import torch

from ..diagonal.identity import fp32_gemm_macro


CHUNK = 64
_BLOCK = 32


def _metadata(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


@lru_cache(maxsize=None)
def _compile_scores(
    key_left_meta,
    key_right_meta,
    query_left_meta,
    query_right_meta,
    scores_meta,
    *,
    length,
    heads,
    left,
    right,
    prior,
    device_index,
):
    import tilelang
    import tilelang.language as T

    gemm = fp32_gemm_macro(T)
    batch = key_left_meta[0][0] // length
    chunks = scores_meta[0][1]

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            KeyLeft: T.StridedTensor(*key_left_meta),
            KeyRight: T.StridedTensor(*key_right_meta),
            QueryLeft: T.StridedTensor(*query_left_meta),
            QueryRight: T.StridedTensor(*query_right_meta),
            Scores: T.StridedTensor(*scores_meta),
        ):
            with T.Kernel(batch * heads, chunks, threads=128) as (batch_head, chunk):
                sequence = batch_head // heads
                head = batch_head % heads
                left_scores = T.alloc_fragment((CHUNK, CHUNK), "float32")
                right_scores = T.alloc_fragment((CHUNK, CHUNK), "float32")
                query_shared = T.alloc_shared((CHUNK, _BLOCK), "float32")
                key_shared = T.alloc_shared((_BLOCK, CHUNK), "float32")
                T.clear(left_scores)
                T.clear(right_scores)
                for block in T.serial(T.ceildiv(left, _BLOCK)):
                    for token, feature in T.Parallel(CHUNK, _BLOCK):
                        index = chunk * CHUNK + token
                        coordinate = block * _BLOCK + feature
                        query_shared[token, feature] = T.if_then_else(
                            (index < length) & (coordinate < left),
                            T.cast(
                                QueryLeft[sequence * length + index, head, coordinate],
                                "float32",
                            ),
                            0.0,
                        )
                    for feature, token in T.Parallel(_BLOCK, CHUNK):
                        index = chunk * CHUNK + token
                        coordinate = block * _BLOCK + feature
                        key_shared[feature, token] = T.if_then_else(
                            (index < length) & (coordinate < left),
                            T.cast(
                                KeyLeft[sequence * length + index, head, coordinate],
                                "float32",
                            ),
                            0.0,
                        )
                    gemm(query_shared, key_shared, left_scores)
                for block in T.serial(T.ceildiv(right, _BLOCK)):
                    for token, feature in T.Parallel(CHUNK, _BLOCK):
                        index = chunk * CHUNK + token
                        coordinate = block * _BLOCK + feature
                        query_shared[token, feature] = T.if_then_else(
                            (index < length) & (coordinate < right),
                            T.cast(
                                QueryRight[sequence * length + index, head, coordinate],
                                "float32",
                            ),
                            0.0,
                        )
                    for feature, token in T.Parallel(_BLOCK, CHUNK):
                        index = chunk * CHUNK + token
                        coordinate = block * _BLOCK + feature
                        key_shared[feature, token] = T.if_then_else(
                            (index < length) & (coordinate < right),
                            T.cast(
                                KeyRight[sequence * length + index, head, coordinate],
                                "float32",
                            ),
                            0.0,
                        )
                    gemm(query_shared, key_shared, right_scores)
                for token, source in T.Parallel(CHUNK, CHUNK):
                    target = chunk * CHUNK + token
                    position = chunk * CHUNK + source
                    causal = (position < target) if prior else (position <= target)
                    Scores[batch_head, chunk, token, source] = T.if_then_else(
                        (target < length) & (position < length) & causal,
                        left_scores[token, source] * right_scores[token, source],
                        0.0,
                    )

        return kernel

    return build_kernel()


def product_chunk_scores(key, query, *, sequence_length, prior=False):
    """Return causal local ProductFeature scores in ``[BH, chunks, 64, 64]``."""
    key_left, key_right = key
    query_left, query_right = query
    if (
        key_left.shape[-1] != query_left.shape[-1]
        or key_right.shape[-1] != query_right.shape[-1]
    ):
        raise ValueError(
            "ProductFeature key and query factors must have matching dimensions"
        )
    if key_left.shape[0] % sequence_length:
        raise ValueError(
            "token-major ProductFeature factors require a uniform sequence length"
        )
    heads = key_left.shape[1]
    chunks = (sequence_length + CHUNK - 1) // CHUNK
    scores = torch.empty(
        key_left.shape[0] // sequence_length * heads,
        chunks,
        CHUNK,
        CHUNK,
        device=key_left.device,
        dtype=torch.float32,
    )
    _compile_scores(
        _metadata(key_left),
        _metadata(key_right),
        _metadata(query_left),
        _metadata(query_right),
        _metadata(scores),
        length=sequence_length,
        heads=heads,
        left=key_left.shape[-1],
        right=key_right.shape[-1],
        prior=prior,
        device_index=key_left.device.index,
    )(key_left, key_right, query_left, query_right, scores)
    return scores


@lru_cache(maxsize=None)
def _compile_readout(
    query_left_meta,
    query_right_meta,
    value_meta,
    carry_meta,
    scores_meta,
    output_meta,
    *,
    length,
    heads,
    left,
    right,
    columns,
    tile_width,
    device_index,
):
    import tilelang
    import tilelang.language as T

    gemm = fp32_gemm_macro(T)
    batch = query_left_meta[0][0] // length
    chunks = carry_meta[0][1]
    product = left * right

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            QueryLeft: T.StridedTensor(*query_left_meta),
            QueryRight: T.StridedTensor(*query_right_meta),
            Value: T.StridedTensor(*value_meta),
            Carry: T.StridedTensor(*carry_meta),
            Scores: T.StridedTensor(*scores_meta),
            Output: T.StridedTensor(*output_meta),
        ):
            with T.Kernel(
                batch * heads, chunks, T.ceildiv(columns, tile_width), threads=128
            ) as (batch_head, chunk, column_tile):
                sequence = batch_head // heads
                head = batch_head % heads
                output = T.alloc_fragment((CHUNK, tile_width), "float32")
                query_product = T.alloc_fragment((CHUNK, _BLOCK), "float32")
                state_shared = T.alloc_shared((_BLOCK, tile_width), "float32")
                scores = T.alloc_fragment((CHUNK, CHUNK), "float32")
                value_shared = T.alloc_shared((CHUNK, tile_width), "float32")
                T.clear(output)
                for product_block in T.serial(T.ceildiv(product, _BLOCK)):
                    for token, feature in T.Parallel(CHUNK, _BLOCK):
                        index = chunk * CHUNK + token
                        product_index = product_block * _BLOCK + feature
                        valid = (index < length) & (product_index < product)
                        query_product[token, feature] = T.if_then_else(
                            valid,
                            T.cast(
                                QueryLeft[
                                    sequence * length + index,
                                    head,
                                    product_index // right,
                                ],
                                "float32",
                            )
                            * T.cast(
                                QueryRight[
                                    sequence * length + index,
                                    head,
                                    product_index % right,
                                ],
                                "float32",
                            ),
                            0.0,
                        )
                    for feature, column in T.Parallel(_BLOCK, tile_width):
                        product_index = product_block * _BLOCK + feature
                        column_index = column_tile * tile_width + column
                        state_shared[feature, column] = T.if_then_else(
                            (product_index < product) & (column_index < columns),
                            T.cast(
                                Carry[batch_head, chunk, product_index, column_index],
                                "float32",
                            ),
                            0.0,
                        )
                    gemm(query_product, state_shared, output)
                for token, source in T.Parallel(CHUNK, CHUNK):
                    scores[token, source] = Scores[batch_head, chunk, token, source]
                for token, column in T.Parallel(CHUNK, tile_width):
                    index = chunk * CHUNK + token
                    column_index = column_tile * tile_width + column
                    value_shared[token, column] = T.if_then_else(
                        (index < length) & (column_index < columns),
                        T.cast(
                            Value[sequence * length + index, head, column_index],
                            "float32",
                        ),
                        0.0,
                    )
                gemm(scores, value_shared, output)
                for token, column in T.Parallel(CHUNK, tile_width):
                    index = chunk * CHUNK + token
                    column_index = column_tile * tile_width + column
                    if (index < length) & (column_index < columns):
                        Output[sequence * length + index, head, column_index] = output[
                            token, column
                        ]

        return kernel

    return build_kernel()


def read_product_state(
    carry, key, query, value, *, sequence_length, prior=False, scores=None
):
    """Contract ProductFeature State using TileLang FP32 GEMMs."""
    key_left, key_right = key
    query_left, query_right = query
    if scores is None:
        scores = product_chunk_scores(
            key, query, sequence_length=sequence_length, prior=prior
        )
    batch_heads, chunks, product, columns = carry.shape
    heads = key_left.shape[1]
    if product != key_left.shape[-1] * key_right.shape[-1]:
        raise ValueError("ProductFeature carry has an incompatible product dimension")
    output = torch.empty(
        key_left.shape[0], heads, columns, device=key_left.device, dtype=torch.float32
    )
    _compile_readout(
        _metadata(query_left),
        _metadata(query_right),
        _metadata(value),
        _metadata(carry),
        _metadata(scores),
        _metadata(output),
        length=sequence_length,
        heads=heads,
        left=key_left.shape[-1],
        right=key_right.shape[-1],
        columns=columns,
        tile_width=16 if columns <= 16 else _BLOCK,
        device_index=key_left.device.index,
    )(query_left, query_right, value, carry, scores, output)
    return output
