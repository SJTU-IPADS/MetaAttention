"""TileLang lowering for vector affine scans over raw inputs."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from ...runtime.bindings import DenseRawBindingRecipe, compile_dense_binding


CHUNK = 64


@dataclass(frozen=True)
class RankOneRecipe:
    state: str
    gate: DenseRawBindingRecipe | None
    value: DenseRawBindingRecipe
    log_gate: bool
    feature: object
    outputs: tuple[tuple[str, torch.dtype], ...]


def compile_rank_one(algorithm):
    """Prove a vector gate/injection and identity readout, without model names."""
    from ....stateful_operator import (
        AxisScale,
        Batch,
        Constant,
        Exp,
        Identity,
        Input,
        Sequence,
        StateTerm,
    )

    if len(algorithm.states) != 1:
        return None
    state = algorithm.states[0]
    if state.rank != 1:
        return None
    feature = state.feature_roles[0]
    transition = algorithm.transitions[0]
    gates = [node for node in transition.propagations if not isinstance(node, Identity)]
    gate = None
    log_gate = False
    specs = {spec.name: spec for spec in algorithm.inputs}
    if gates:
        if len(gates) != 1 or not isinstance(gates[0], AxisScale):
            return None
        if gates[0].roles != (feature,):
            return None
        expression = gates[0].factor
        log_gate = isinstance(expression, Exp)
        expression = expression.operand if log_gate else expression
        if not isinstance(expression, Input):
            return None
        gate = specs[expression.name]
    if len(transition.injections) != 1:
        return None
    factors = transition.injections[0].factors
    if len(factors) != 1 or factors[0].roles != (feature,):
        return None
    if not isinstance(factors[0].expression, Input):
        return None
    value = specs[factors[0].expression.name]
    if any(
        Batch not in spec.roles or Sequence not in spec.roles
        for spec in ((value, gate) if gate is not None else (value,))
    ):
        return None
    outputs = []
    for readout in algorithm.readouts:
        term = getattr(readout, "expression", None)
        if not isinstance(term, StateTerm) or term.state != state.name or term.prior:
            return None
        if term.role is not None or not isinstance(term.operand, Constant):
            return None
        if term.operand.value != 1.0:
            return None
        outputs.append((readout.name, readout.output_dtype))
    if len(outputs) != 1:
        return None
    return RankOneRecipe(
        state.name,
        compile_dense_binding(gate, feature) if gate is not None else None,
        compile_dense_binding(value, feature),
        log_gate,
        feature,
        tuple(outputs),
    )


def _storage_view(tensor: torch.Tensor):
    element_size = tensor.element_size()
    storage_elements = tensor.untyped_storage().nbytes() // element_size
    return tensor.as_strided(
        (storage_elements - tensor.storage_offset(),), (1,), tensor.storage_offset()
    )


def _metadata(tensor: torch.Tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
    )


@lru_cache(maxsize=None)
def _compile_summary(
    value_meta,
    gate_meta,
    scale_meta,
    bias_meta,
    *,
    batch,
    length,
    heads,
    features,
    value_strides,
    gate_strides,
    value_group,
    gate_group,
    log_gate,
    identity,
    device_index,
):
    import tilelang
    import tilelang.language as T

    chunks = tilelang.cdiv(length, CHUNK)

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Value: T.StridedTensor(*value_meta),
            Gate: T.StridedTensor(*gate_meta),
            Scale: T.StridedTensor(*scale_meta),
            Bias: T.StridedTensor(*bias_meta),
        ):
            with T.Kernel(
                batch * heads, chunks, T.ceildiv(features, 128), threads=128
            ) as (sh, chunk, block):
                sequence = sh // heads
                head = sh % heads
                scale = T.alloc_fragment((128,), "float32")
                bias = T.alloc_fragment((128,), "float32")
                for feature in T.Parallel(128):
                    x = block * 128 + feature
                    scale[feature] = 1.0
                    bias[feature] = 0.0
                    for token in T.serial(CHUNK):
                        logical = chunk * CHUNK + token
                        if logical < length and x < features:
                            value = T.cast(
                                Value[
                                    sequence * value_strides[0]
                                    + (head // value_group) * value_strides[1]
                                    + logical * value_strides[2]
                                    + x * value_strides[3]
                                ],
                                "float32",
                            )
                            if identity:
                                gate = 1.0
                            else:
                                gate = T.cast(
                                    Gate[
                                        sequence * gate_strides[0]
                                        + (head // gate_group) * gate_strides[1]
                                        + logical * gate_strides[2]
                                        + x * gate_strides[3]
                                    ],
                                    "float32",
                                )
                                if log_gate:
                                    gate = T.exp(gate)
                            bias[feature] = gate * bias[feature] + value
                            scale[feature] = gate * scale[feature]
                for feature in T.Parallel(128):
                    x = block * 128 + feature
                    if x < features:
                        Scale[sh, chunk, x] = scale[feature]
                        Bias[sh, chunk, x] = bias[feature]

        return kernel

    return build_kernel()


@lru_cache(maxsize=None)
def _compile_apply(
    value_meta,
    gate_meta,
    initial_meta,
    output_meta,
    final_meta,
    carry_meta,
    *,
    batch,
    length,
    heads,
    features,
    value_strides,
    gate_strides,
    value_group,
    gate_group,
    log_gate,
    identity,
    chunks,
    write_final,
    device_index,
):
    import tilelang
    import tilelang.language as T

    @tilelang.jit(compile_flags=["-fmad=false"])
    def build_kernel():
        @T.prim_func
        def kernel(
            Value: T.StridedTensor(*value_meta),
            Gate: T.StridedTensor(*gate_meta),
            Initial: T.StridedTensor(*initial_meta),
            Carry: T.StridedTensor(*carry_meta),
            Output: T.StridedTensor(*output_meta),
            Final: T.StridedTensor(*final_meta),
        ):
            with T.Kernel(
                batch * heads, chunks, T.ceildiv(features, 128), threads=128
            ) as (sh, chunk, block):
                sequence = sh // heads
                head = sh % heads
                state = T.alloc_fragment((128,), "float32")
                for feature in T.Parallel(128):
                    x = block * 128 + feature
                    if x < features:
                        state[feature] = T.cast(Carry[sh, chunk, x], "float32")
                        for token in T.serial(CHUNK):
                            logical = chunk * CHUNK + token
                            if logical < length:
                                value = T.cast(
                                    Value[
                                        sequence * value_strides[0]
                                        + (head // value_group) * value_strides[1]
                                        + logical * value_strides[2]
                                        + x * value_strides[3]
                                    ],
                                    "float32",
                                )
                                if identity:
                                    gate = 1.0
                                else:
                                    gate = T.cast(
                                        Gate[
                                            sequence * gate_strides[0]
                                            + (head // gate_group) * gate_strides[1]
                                            + logical * gate_strides[2]
                                            + x * gate_strides[3]
                                        ],
                                        "float32",
                                    )
                                    if log_gate:
                                        gate = T.exp(gate)
                                state[feature] = gate * state[feature] + value
                                Output[sequence, head, logical, x] = state[feature]
                        if write_final and chunk == chunks - 1:
                            Final[sequence, head, x] = state[feature]

        return kernel

    return build_kernel()


def execute_rank_one(
    recipe,
    inputs,
    initial,
    *,
    sequence_count,
    sequence_length,
    heads,
    return_final_state,
):
    value, vs, vg = recipe.value.bind(inputs[recipe.value.name], heads)
    value = _storage_view(value)
    if recipe.gate is None:
        gate, gs, gg = _storage_view(initial), (0, 0, 0, 0), 1
        identity = True
    else:
        gate, gs, gg = recipe.gate.bind(inputs[recipe.gate.name], heads)
        gate = _storage_view(gate)
        identity = False
    features = initial.shape[-1]
    chunks = (sequence_length + CHUNK - 1) // CHUNK
    output = torch.empty(
        sequence_count,
        heads,
        sequence_length,
        features,
        device=initial.device,
        dtype=recipe.outputs[0][1],
    )
    if sequence_length == 0:
        return output, initial.clone() if return_final_state else None
    final = torch.empty_like(initial) if return_final_state else initial
    device_index = (
        initial.device.index
        if initial.device.index is not None
        else torch.cuda.current_device()
    )
    if chunks == 1:
        carry = initial.reshape(sequence_count * heads, 1, features)
    else:
        shape = (sequence_count * heads, chunks, features)
        scale = torch.empty(shape, device=initial.device, dtype=torch.float32)
        bias = torch.empty_like(scale)
        carry = torch.empty_like(scale)
        _compile_summary(
            _metadata(value),
            _metadata(gate),
            _metadata(scale),
            _metadata(bias),
            batch=sequence_count,
            length=sequence_length,
            heads=heads,
            features=features,
            value_strides=vs,
            gate_strides=gs,
            value_group=vg,
            gate_group=gg,
            log_gate=recipe.log_gate,
            identity=identity,
            device_index=device_index,
        )(value, gate, scale, bias)
        from ..diagonal.diagonal_scan import hierarchical_carry

        hierarchical_carry(
            scale.reshape(sequence_count * heads, chunks, 1, features),
            bias.reshape(sequence_count * heads, chunks, 1, features),
            initial.reshape(sequence_count * heads, 1, features),
            carry.reshape(sequence_count * heads, chunks, 1, features),
            final.reshape(sequence_count * heads, 1, features),
            identity=False,
            write_final=False,
        )
    _compile_apply(
        _metadata(value),
        _metadata(gate),
        _metadata(initial),
        _metadata(output),
        _metadata(final),
        _metadata(carry),
        batch=sequence_count,
        length=sequence_length,
        heads=heads,
        features=features,
        value_strides=vs,
        gate_strides=gs,
        value_group=vg,
        gate_group=gg,
        log_gate=recipe.log_gate,
        identity=identity,
        chunks=chunks,
        write_final=return_final_state,
        device_index=device_index,
    )(value, gate, initial, carry, output, final)
    return output, final if return_final_state else None
