"""TileLang exclusive prefixes for identity chunk summaries."""

from functools import lru_cache

import torch


_BLOCK = 256
_GROUP = 32


def _metadata(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


def _state_view(state):
    # Public states may retain separate batch/head axes or already combine them.
    if state.ndim == 3:
        return torch.as_strided(
            state,
            (state.shape[0], 1, *state.shape[-2:]),
            (state.stride(0), *state.stride()),
        )
    return state


@lru_cache(maxsize=None)
def _compile_identity_scan(
    bias_meta,
    initial_meta,
    carry_meta,
    final_meta,
    den_bias_meta,
    den_initial_meta,
    den_carry_meta,
    den_final_meta,
    mode: str,
    write_final: bool,
    device_index: int,
):
    import tilelang
    import tilelang.language as T

    paired = den_bias_meta is not None
    bh, chunks, rows, columns = bias_meta[0]
    elements = rows * columns
    blocks = (elements + _BLOCK - 1) // _BLOCK
    den_elements = den_bias_meta[0][2] * den_bias_meta[0][3] if paired else 0
    grid_blocks = blocks + (den_elements + _BLOCK - 1) // _BLOCK
    groups = (chunks + _GROUP - 1) // _GROUP if mode != "direct" else 1
    steps = chunks if mode == "direct" else _GROUP
    # Unroll direct levels so aliasing summaries/carries have disjoint addresses.
    chunk_loop = T.unroll if paired and mode == "direct" else T.serial
    chunk_annotations = (
        {"pragma_unroll_explicit": True} if paired and mode == "direct" else {}
    )

    @T.macro
    def scan(Bias, Initial, Carry, Final, sh, group, block):
        state = T.alloc_fragment((_BLOCK,), "float32")
        summary = T.alloc_fragment((_BLOCK,), "float32")
        cols = Bias.shape[3]
        size = Bias.shape[2] * cols
        for lane in T.Parallel(_BLOCK):
            x = block * _BLOCK + lane
            if mode == "local":
                state[lane] = 0.0
            elif mode == "direct":
                state[lane] = T.if_then_else(
                    x < size,
                    Initial[
                        sh // Initial.shape[1],
                        sh % Initial.shape[1],
                        x // cols,
                        x % cols,
                    ],
                    0.0,
                )
            else:
                state[lane] = T.if_then_else(
                    x < size,
                    Initial[sh, group, x // cols, x % cols],
                    0.0,
                )
        for step in chunk_loop(steps, annotations=chunk_annotations):
            chunk = step if mode == "direct" else group * _GROUP + step
            # Consume the summary before an in-place exclusive-carry store.
            for lane in T.Parallel(_BLOCK):
                x = block * _BLOCK + lane
                summary[lane] = T.if_then_else(
                    (chunk < chunks) & (x < size),
                    Bias[sh, chunk, x // cols, x % cols],
                    0.0,
                )
            for lane in T.Parallel(_BLOCK):
                x = block * _BLOCK + lane
                if mode != "local":
                    if (chunk < chunks) & (x < size):
                        Carry[sh, chunk, x // cols, x % cols] = state[lane]
                state[lane] += summary[lane]
        for lane in T.Parallel(_BLOCK):
            x = block * _BLOCK + lane
            if x < size:
                if mode == "local":
                    Carry[sh, group, x // cols, x % cols] = state[lane]
                elif write_final:
                    if group == groups - 1:
                        Final[
                            sh // Final.shape[1],
                            sh % Final.shape[1],
                            x // cols,
                            x % cols,
                        ] = state[lane]

    @tilelang.jit
    def build_kernel():
        if paired:

            @T.prim_func
            def kernel(
                Bias: T.StridedTensor(*bias_meta),
                Initial: T.StridedTensor(*initial_meta),
                Carry: T.StridedTensor(*carry_meta),
                Final: T.StridedTensor(*final_meta),
                DenBias: T.StridedTensor(*den_bias_meta),
                DenInitial: T.StridedTensor(*den_initial_meta),
                DenCarry: T.StridedTensor(*den_carry_meta),
                DenFinal: T.StridedTensor(*den_final_meta),
            ):
                with T.Kernel(bh, groups, grid_blocks, threads=128) as (
                    sh,
                    group,
                    block,
                ):
                    if block < blocks:
                        scan(Bias, Initial, Carry, Final, sh, group, block)
                    else:
                        scan(
                            DenBias,
                            DenInitial,
                            DenCarry,
                            DenFinal,
                            sh,
                            group,
                            block - blocks,
                        )
        else:

            @T.prim_func
            def kernel(
                Bias: T.StridedTensor(*bias_meta),
                Initial: T.StridedTensor(*initial_meta),
                Carry: T.StridedTensor(*carry_meta),
                Final: T.StridedTensor(*final_meta),
            ):
                with T.Kernel(bh, groups, blocks, threads=128) as (sh, group, block):
                    scan(Bias, Initial, Carry, Final, sh, group, block)

        return kernel

    return build_kernel()


def _launch(mode, bias, initial, carry, final, *, write_final, pair=None):
    initial = _state_view(initial) if mode != "apply" else initial
    final = _state_view(final) if final is not None else initial
    tensors = (bias, initial, carry, final)
    if pair is None:
        den_meta = (None,) * 4
    else:
        den_bias, den_initial, den_carry, den_final = pair
        den_initial = _state_view(den_initial) if mode != "apply" else den_initial
        den_final = _state_view(den_final) if den_final is not None else den_initial
        pair = (den_bias, den_initial, den_carry, den_final)
        den_meta = tuple(_metadata(tensor) for tensor in pair)
    device_index = bias.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel = _compile_identity_scan(
        *(_metadata(tensor) for tensor in tensors),
        *den_meta,
        mode,
        write_final,
        device_index,
    )
    kernel(*tensors, *(pair or ()))


def identity_hierarchical_carry(bias, initial, carry, final, *, write_final):
    """Write exclusive FP32 carries without changing initial or chunk summaries."""
    bh, chunks, rows, columns = bias.shape
    if chunks == 0 and not write_final:
        return
    if chunks <= 64:
        _launch("direct", bias, initial, carry, final, write_final=write_final)
        return
    groups = (chunks + _GROUP - 1) // _GROUP
    group_bias = torch.empty(
        (bh, groups, rows, columns), device=bias.device, dtype=torch.float32
    )
    group_carry = torch.empty_like(group_bias)
    _launch("local", bias, initial, group_bias, final, write_final=False)
    identity_hierarchical_carry(
        group_bias, initial, group_carry, final, write_final=False
    )
    _launch("apply", bias, group_carry, carry, final, write_final=write_final)


def identity_hierarchical_carry_pair(
    bias,
    den_bias,
    initial,
    den_initial,
    final,
    den_final,
    *,
    write_final,
):
    """Pair every level; direct levels reuse summaries after consuming each bias."""
    bh, chunks, rows, columns = bias.shape
    if chunks == 0 and not write_final:
        return bias, den_bias
    if chunks <= 64:
        _launch(
            "direct",
            bias,
            initial,
            bias,
            final,
            write_final=write_final,
            pair=(den_bias, den_initial, den_bias, den_final),
        )
        return bias, den_bias
    groups = (chunks + _GROUP - 1) // _GROUP
    group_bias = torch.empty(
        (bh, groups, rows, columns), device=bias.device, dtype=torch.float32
    )
    den_group_bias = torch.empty(
        (bh, groups, *den_bias.shape[-2:]),
        device=den_bias.device,
        dtype=torch.float32,
    )
    # All original biases are consumed before the direct recursion overwrites
    # its group summaries. Long applies keep their input and output separate.
    _launch(
        "local",
        bias,
        initial,
        group_bias,
        final,
        write_final=False,
        pair=(den_bias, den_initial, den_group_bias, den_final),
    )
    group_carry, den_group_carry = identity_hierarchical_carry_pair(
        group_bias,
        den_group_bias,
        initial,
        den_initial,
        final,
        den_final,
        write_final=False,
    )
    carry, den_carry = torch.empty_like(bias), torch.empty_like(den_bias)
    _launch(
        "apply",
        bias,
        group_carry,
        carry,
        final,
        write_final=write_final,
        pair=(den_bias, den_group_carry, den_carry, den_final),
    )
    return carry, den_carry
