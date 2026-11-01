"""TileLang hierarchical affine prefixes for diagonal chunk summaries."""

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
    """Expose a vector State as a one-group 4-D view without copying."""
    if state.ndim == 3:
        return torch.as_strided(
            state,
            (state.shape[0], 1, *state.shape[-2:]),
            (state.stride(0), *state.stride()),
        )
    return state


@lru_cache(maxsize=None)
def _compile_affine_scan(
    scale_meta,
    bias_meta,
    initial_meta,
    carry_meta,
    final_meta,
    mode: str,
    write_final: bool,
    device_index: int,
):
    import tilelang
    import tilelang.language as T

    bh, chunks, rows, columns = bias_meta[0]
    elements = rows * columns
    blocks = (elements + _BLOCK - 1) // _BLOCK
    groups = (chunks + _GROUP - 1) // _GROUP if mode != "direct" else 1
    steps = chunks if mode == "direct" else _GROUP

    @T.macro
    def scan(Scale, Bias, Initial, Carry, Final, sh, group, block):
        if mode == "local":
            group_scale = T.alloc_fragment((_BLOCK,), "float32")
            group_bias = T.alloc_fragment((_BLOCK,), "float32")
            for lane in T.Parallel(_BLOCK):
                group_scale[lane] = 1.0
                group_bias[lane] = 0.0
            for step in T.serial(steps):
                chunk = group * _GROUP + step
                for lane in T.Parallel(_BLOCK):
                    x = block * _BLOCK + lane
                    valid = (chunk < chunks) & (x < elements)
                    scale = T.if_then_else(
                        valid, Scale[sh, chunk, x // columns, x % columns], 1.0
                    )
                    bias = T.if_then_else(
                        valid, Bias[sh, chunk, x // columns, x % columns], 0.0
                    )
                    group_bias[lane] = scale * group_bias[lane] + bias
                    group_scale[lane] = scale * group_scale[lane]
            for lane in T.Parallel(_BLOCK):
                x = block * _BLOCK + lane
                if x < elements:
                    Carry[sh, group, x // columns, x % columns] = group_scale[lane]
                    Final[sh, group, x // columns, x % columns] = group_bias[lane]
            return

        state = T.alloc_fragment((_BLOCK,), "float32")
        for lane in T.Parallel(_BLOCK):
            x = block * _BLOCK + lane
            if mode == "direct":
                state[lane] = T.if_then_else(
                    x < elements,
                    Initial[
                        sh // Initial.shape[1],
                        sh % Initial.shape[1],
                        x // columns,
                        x % columns,
                    ],
                    0.0,
                )
            else:
                state[lane] = T.if_then_else(
                    x < elements,
                    Initial[sh, group, x // columns, x % columns],
                    0.0,
                )
        for step in T.serial(steps):
            chunk = step if mode == "direct" else group * _GROUP + step
            for lane in T.Parallel(_BLOCK):
                x = block * _BLOCK + lane
                valid = (chunk < chunks) & (x < elements)
                scale = T.if_then_else(
                    valid, Scale[sh, chunk, x // columns, x % columns], 1.0
                )
                bias = T.if_then_else(
                    valid, Bias[sh, chunk, x // columns, x % columns], 0.0
                )
                if valid:
                    Carry[sh, chunk, x // columns, x % columns] = state[lane]
                state[lane] = scale * state[lane] + bias
        if write_final:
            for lane in T.Parallel(_BLOCK):
                x = block * _BLOCK + lane
                if x < elements and (mode == "direct" or group == groups - 1):
                    Final[
                        sh // Final.shape[1],
                        sh % Final.shape[1],
                        x // columns,
                        x % columns,
                    ] = state[lane]

    scale_shape, scale_strides, scale_dtype = scale_meta
    bias_shape, bias_strides, bias_dtype = bias_meta
    initial_shape, initial_strides, initial_dtype = initial_meta
    carry_shape, carry_strides, carry_dtype = carry_meta
    final_shape, final_strides, final_dtype = final_meta

    @tilelang.jit
    def build_kernel():
        if mode == "local":

            @T.prim_func
            def kernel(
                Scale: T.StridedTensor(scale_shape, scale_strides, scale_dtype),
                Bias: T.StridedTensor(bias_shape, bias_strides, bias_dtype),
                GroupScale: T.StridedTensor(
                    initial_shape, initial_strides, initial_dtype
                ),
                GroupBias: T.StridedTensor(carry_shape, carry_strides, carry_dtype),
            ):
                with T.Kernel(bh, groups, blocks, threads=128) as (sh, group, block):
                    scan(Scale, Bias, None, GroupScale, GroupBias, sh, group, block)
        else:

            @T.prim_func
            def kernel(
                Scale: T.StridedTensor(scale_shape, scale_strides, scale_dtype),
                Bias: T.StridedTensor(bias_shape, bias_strides, bias_dtype),
                Initial: T.StridedTensor(initial_shape, initial_strides, initial_dtype),
                Carry: T.StridedTensor(carry_shape, carry_strides, carry_dtype),
                Final: T.StridedTensor(final_shape, final_strides, final_dtype),
            ):
                with T.Kernel(bh, groups, blocks, threads=128) as (sh, group, block):
                    scan(Scale, Bias, Initial, Carry, Final, sh, group, block)

        return kernel

    return build_kernel()


def _launch(mode, scale, bias, initial, carry, final, *, write_final):
    initial = _state_view(initial) if mode != "apply" else initial
    final = _state_view(final) if final is not None else initial
    device_index = bias.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if mode == "local":
        kernel = _compile_affine_scan(
            _metadata(scale),
            _metadata(bias),
            _metadata(initial),
            _metadata(carry),
            _metadata(carry),
            mode,
            write_final,
            device_index,
        )
        kernel(scale, bias, initial, carry)
        return
    kernel = _compile_affine_scan(
        _metadata(scale),
        _metadata(bias),
        _metadata(initial),
        _metadata(carry),
        _metadata(final),
        mode,
        write_final,
        device_index,
    )
    kernel(scale, bias, initial, carry, final)


def hierarchical_carry(scale, bias, initial, carry, final, *, identity, write_final):
    """Prefix affine chunk summaries without division or inverse reconstruction."""
    if identity:
        from .identity_scan import identity_hierarchical_carry

        return identity_hierarchical_carry(
            bias,
            initial,
            carry,
            final,
            write_final=write_final,
        )
    _, chunks, rows, columns = bias.shape
    if chunks <= 64:
        _launch("direct", scale, bias, initial, carry, final, write_final=write_final)
        return
    groups = (chunks + _GROUP - 1) // _GROUP
    group_scale = torch.empty(
        (bias.shape[0], groups, rows, columns),
        device=bias.device,
        dtype=torch.float32,
    )
    group_bias = torch.empty_like(group_scale)
    group_carry = torch.empty_like(group_scale)
    _launch(
        "local", scale, bias, group_scale, group_bias, group_bias, write_final=False
    )
    hierarchical_carry(
        group_scale,
        group_bias,
        initial,
        group_carry,
        final,
        identity=False,
        write_final=False,
    )
    _launch("apply", scale, bias, group_carry, carry, final, write_final=write_final)
