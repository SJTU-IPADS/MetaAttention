from __future__ import annotations

import os
from typing import Mapping

import torch

from ...stateful_operator import (
    Batch,
    FeatureRole,
    HeadRole,
    LogicalRole,
    Sequence,
    StateTuple,
    StatefulCompilationError,
    Target,
    _InputValidation,
    _Runtime,
    _SingletonRole,
    _StateValidation,
    _ValidationRecipe,
    _identity,
    _verify_product_roles,
)


# Successful entries only: unsupported hardware and constraint failures must be
# rechecked and continue to raise their existing structured errors.
_VALIDATED_TARGETS: set[tuple[int, str | None, str | None, tuple[int, int] | None]] = (
    set()
)


def _validated_target_key(index: int, target):
    """Return a hashable key without changing malformed-target behavior."""
    if target is None:
        return (index, None, None, None)
    device_type = getattr(target, "device_type", None)
    product_family = getattr(target, "product_family", None)
    capability = getattr(target, "capability", None)
    if not (
        isinstance(device_type, str)
        and isinstance(product_family, str)
        and isinstance(capability, tuple)
        and len(capability) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) for item in capability
        )
    ):
        return None
    return (
        index,
        device_type,
        product_family.upper(),
        capability,
    )


def validate_input_names(
    expected_names: tuple[str, ...], inputs: Mapping[str, torch.Tensor]
) -> None:
    missing = [name for name in expected_names if name not in inputs]
    unknown = [name for name in inputs if name not in expected_names]
    if missing:
        raise StatefulCompilationError(
            "RUNTIME_BINDING", "inputs", {"missing": sorted(missing)}
        )
    if unknown:
        raise StatefulCompilationError(
            "RUNTIME_BINDING", "inputs", {"unknown": sorted(unknown)}
        )


def validate_target(device: torch.device, target) -> None:
    if device.type == "cpu" and os.environ.get("STATEFUL_OPERATOR_TEST_CPU") == "1":
        return
    if device.type != "cuda" or torch.version.hip is not None:
        raise StatefulCompilationError(
            "TARGET", "target", {"expected": "CUDA H20", "actual": device.type}
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    target_key = _validated_target_key(index, target)
    if target_key is not None and target_key in _VALIDATED_TARGETS:
        return
    product = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    if "MIG" in product.upper() or "H20" not in product.upper() or capability != (9, 0):
        raise StatefulCompilationError(
            "TARGET",
            "target",
            {
                "expected_product": "H20",
                "actual_product": product,
                "capability": capability,
            },
        )
    if target is not None and (
        target.device_type != "cuda"
        or target.product_family.upper() not in product.upper()
        or target.capability != capability
    ):
        from ...stateful_operator import _canonical

        raise StatefulCompilationError(
            "TARGET_CONSTRAINT", "target", {"constraint": _canonical(target)}
        )
    if target_key is not None:
        _VALIDATED_TARGETS.add(target_key)


def validate_aliases(inputs: Mapping[str, torch.Tensor]) -> None:
    # Public layout validation has already established contiguous tensors, so
    # each input occupies one byte interval. Read storage metadata once per
    # input, not twice for every pair; still inspect current pointers each call.
    tensors = tuple(
        (name, tensor, tensor.untyped_storage().data_ptr())
        for name, tensor in inputs.items()
    )
    intervals = {}
    for index, (left_name, left, left_pointer) in enumerate(tensors):
        for right_name, right, right_pointer in tensors[index + 1 :]:
            if left is right or left_pointer != right_pointer:
                continue
            for name, tensor in ((left_name, left), (right_name, right)):
                if name not in intervals:
                    itemsize = tensor.element_size()
                    start = tensor.storage_offset() * itemsize
                    intervals[name] = (start, start + tensor.numel() * itemsize)
            left_start, left_end = intervals[left_name]
            right_start, right_end = intervals[right_name]
            if max(left_start, right_start) < min(left_end, right_end):
                raise StatefulCompilationError(
                    "RUNTIME_ALIAS", "inputs", {"overlapping": [left_name, right_name]}
                )


def validate_compile_roles(program, role_sizes, target, mode):
    """Validate the fixed-role contract without assuming a scan algebra."""
    if not isinstance(target, Target):
        raise StatefulCompilationError("COMPILE_TARGET", "target", {})
    if mode not in ("dense", "packed"):
        raise StatefulCompilationError(
            "COMPILE_MODE", "mode", {"expected": ["dense", "packed"]}
        )
    declared = {
        role
        for spec in program.inputs
        for role in spec.roles
        if isinstance(role, (HeadRole, FeatureRole))
    }
    declared.update(
        role
        for state in program.states
        for role in state.roles
        if isinstance(role, (HeadRole, FeatureRole))
    )
    actual = set(role_sizes)
    invalid = [role for role in actual if not isinstance(role, (HeadRole, FeatureRole))]
    if invalid:
        raise StatefulCompilationError(
            "COMPILE_ROLE_SIZE", "role_sizes", {"runtime_roles": invalid}
        )
    missing = declared - actual
    unknown = actual - declared
    if missing or unknown:
        raise StatefulCompilationError(
            "COMPILE_ROLE_SIZE",
            "role_sizes",
            {
                "missing": sorted(map(str, missing)),
                "unknown": sorted(map(str, unknown)),
            },
        )
    for role, size in role_sizes.items():
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise StatefulCompilationError(
                "COMPILE_ROLE_SIZE", f"role_sizes.{role}", {"size": size}
            )
    head_roles = {spec.head_role for spec in program.states}
    if len(head_roles) != 1:
        raise StatefulCompilationError(
            "IR_SCHEMA",
            "states",
            {
                "expected": "States share one Head role",
                "actual": sorted(str(role) for role in head_roles),
            },
        )
    _verify_product_roles(dict(role_sizes), "role_sizes")

    return tuple(sorted(role_sizes.items(), key=lambda item: _identity(item[0])))


def compile_validation(
    program,
    *,
    reference_required,
    native_diagonal=False,
    state_identity=0.0,
    allow_empty_dense=False,
):
    """Freeze tensor declarations and the algebra's invocation semantics."""
    return _ValidationRecipe(
        expected_input_names=tuple(spec.name for spec in program.inputs),
        state_names=tuple(spec.name for spec in program.states),
        output_names=tuple(readout.name for readout in program.readouts),
        inputs=tuple(
            _InputValidation(
                spec.name,
                spec.roles,
                spec.dtype,
                spec.differentiable,
                spec.roles.index(Sequence) if Sequence in spec.roles else None,
                tuple(role for role in spec.roles if role not in (Batch, Sequence)),
            )
            for spec in program.inputs
        ),
        states=tuple(
            _StateValidation(
                spec.name, spec.head_role, spec.feature_roles, state_identity
            )
            for spec in program.states
        ),
        reference_required=reference_required,
        native_diagonal=native_diagonal,
        head_mapping=program.head_mapping.mappings,
        allow_empty_dense=allow_empty_dense,
    )


def validate_call(
    inputs,
    *,
    initial_state,
    sequence_offsets,
    role_sizes,
    validation,
    mode,
    target,
    backward,
    default_states=None,
    return_final_state=False,
):
    """Validate a public invocation before resource checks or execution.

    State defaults and dense-empty legality belong to the compiled contract,
    not to whichever operator happens to validate a composed scan first.
    Only affine calls pass an optional cache for unobservable default States.
    """
    validate_input_names(validation.expected_input_names, inputs)
    runtime, sizes = _validate_runtime(inputs, sequence_offsets, role_sizes, validation)
    if runtime.packed != (mode == "packed"):
        raise StatefulCompilationError(
            "RUNTIME_MODE", "sequence_offsets", {"expected": mode}
        )
    validate_target(runtime.device, target)
    validate_aliases(inputs)
    if runtime.token_count == 0 and return_final_state:
        default_states = None
    states = _validate_state(
        initial_state, runtime, sizes, validation, default_states=default_states
    )
    requires_grad = any(t.requires_grad for t in (*inputs.values(), *states.values()))
    if not backward and torch.is_grad_enabled() and requires_grad:
        raise StatefulCompilationError(
            "RUNTIME_GRAD", "inputs", {"backward": "not compiled"}
        )
    return runtime, sizes, states, requires_grad


def _validate_runtime(
    inputs: Mapping[str, torch.Tensor],
    offsets: torch.Tensor | None,
    declared_roles: tuple[tuple[LogicalRole, int], ...],
    validation: _ValidationRecipe,
) -> tuple[_Runtime, dict[LogicalRole, int]]:
    packed = offsets is not None
    device: torch.device | None = None
    # Fixed Roles are validated against the compiled contract; a Role carried
    # only by a State (never by an Input) must still resolve.
    # Compilation already restricts these immutable pairs to Head/Feature
    # Roles. Make one mutable extent map; all input checks below still run.
    role_sizes: dict[LogicalRole, int] = dict(declared_roles)
    token_count: int | None = None
    uniform_length: int | None = None
    for spec in validation.inputs:
        tensor = inputs[spec.name]
        path = f"inputs.{spec.name}"
        if not isinstance(tensor, torch.Tensor):
            raise StatefulCompilationError(
                "RUNTIME_BINDING", path, {"expected": "torch.Tensor"}
            )
        if tensor.dtype != spec.dtype:
            raise StatefulCompilationError(
                "RUNTIME_DTYPE",
                path,
                {"expected": str(spec.dtype), "actual": str(tensor.dtype)},
            )
        if not tensor.is_contiguous():
            raise StatefulCompilationError(
                "RUNTIME_LAYOUT", path, {"expected": "contiguous"}
            )
        if not spec.differentiable and tensor.requires_grad:
            raise StatefulCompilationError(
                "RUNTIME_GRAD", path, {"differentiable": False}
            )
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise StatefulCompilationError("RUNTIME_DEVICE", "inputs", {})
        if packed and spec.sequence_axis is None:
            raise StatefulCompilationError(
                "SPECIALIZATION",
                path,
                {"reason": "packed non-sequence inputs are not supported"},
            )
        if packed and spec.sequence_axis is not None:
            physical_roles = spec.packed_roles
            if tensor.ndim != len(physical_roles) + 1:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE", path, {"mode": "packed"}
                )
            token_count = tensor.shape[0] if token_count is None else token_count
            if tensor.shape[0] != token_count:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE", path, {"role": "sequence"}
                )
            dimensions = zip(physical_roles, tensor.shape[1:])
        else:
            if tensor.ndim != len(spec.roles):
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE", path, {"expected_ndim": len(spec.roles)}
                )
            dimensions = zip(spec.roles, tensor.shape)
            if spec.sequence_axis is not None:
                length = tensor.shape[spec.sequence_axis]
                uniform_length = length if uniform_length is None else uniform_length
                if length != uniform_length or (
                    length == 0 and not validation.allow_empty_dense
                ):
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"role": "sequence"}
                    )
        for role, size in dimensions:
            if isinstance(role, _SingletonRole):
                if role not in (Batch, Sequence) and size != 1:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", path, {"role": str(role), "expected": 1}
                    )
                if role not in (Batch, Sequence):
                    continue
            previous = role_sizes.setdefault(role, size)
            if previous != size:
                raise StatefulCompilationError(
                    "RUNTIME_SHAPE",
                    path,
                    {"role": role, "sizes": [previous, size]},
                )
    assert device is not None
    if packed:
        assert offsets is not None
        token_count = token_count or 0
        if (
            offsets.dtype != torch.int32
            or offsets.ndim != 1
            or not offsets.is_contiguous()
            or offsets.device != device
            or offsets.numel() < 1
        ):
            raise StatefulCompilationError(
                "RUNTIME_OFFSETS",
                "sequence_offsets",
                {"expected": "device-local contiguous int32 vector"},
            )
        torch._assert_async(offsets[0] == 0, "sequence_offsets must start at zero")
        torch._assert_async(
            offsets[-1] == token_count,
            "sequence_offsets must end at total token count",
        )
        torch._assert_async(
            torch.all(offsets[1:] >= offsets[:-1]),
            "sequence_offsets must be nondecreasing",
        )
        sequence_count = offsets.shape[0] - 1
        role_sizes[Batch] = sequence_count
    else:
        if uniform_length is None:
            raise StatefulCompilationError(
                "RUNTIME_SHAPE", "inputs", {"sequence": "required"}
            )
        sequence_count = role_sizes[Batch]
        token_count = sequence_count * uniform_length
    state_spec = validation.states[0]
    try:
        state_heads = role_sizes[state_spec.head_role]
        feature_sizes = tuple(role_sizes[role] for role in state_spec.feature_roles)
    except KeyError as error:
        raise StatefulCompilationError(
            "RUNTIME_SHAPE", "states[0].roles", {"missing_role": error.args[0]}
        ) from error
    for spec in validation.states:
        try:
            sizes = tuple(role_sizes[role] for role in spec.feature_roles)
        except KeyError as error:
            raise StatefulCompilationError(
                "RUNTIME_SHAPE",
                f"states.{spec.name}",
                {"missing_role": error.args[0]},
            ) from error
        if (
            device.type == "cuda"
            and not validation.reference_required
            and not validation.native_diagonal
            and any(size not in (64, 128) for size in sizes)
        ):
            raise StatefulCompilationError(
                "SPECIALIZATION",
                "states[0].roles",
                {
                    "state": spec.name,
                    "feature_sizes": sizes,
                    "supported": [64, 128],
                },
            )
    _verify_product_roles(role_sizes, "inputs")
    for source, target in validation.head_mapping:
        if target != state_spec.head_role or source not in role_sizes:
            raise StatefulCompilationError(
                "IR_TYPE", "head_mapping", {"source": source, "target": target}
            )
        if state_heads % role_sizes[source]:
            raise StatefulCompilationError(
                "RUNTIME_SHAPE", "head_mapping", {"source": source}
            )
    return (
        _Runtime(
            packed,
            device,
            sequence_count,
            token_count,
            uniform_length,
            state_heads,
            tuple(feature_sizes),
            offsets,
        ),
        role_sizes,
    )


def _validate_state(
    state: StateTuple | None,
    runtime: _Runtime,
    role_sizes: Mapping[LogicalRole, int],
    validation: _ValidationRecipe,
    *,
    default_states: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Validate initial State for every declared State and return them by name."""
    if isinstance(state, StateTuple) and any(
        not isinstance(tensor, torch.Tensor) for tensor in state
    ):
        raise StatefulCompilationError(
            "RUNTIME_STATE", "initial_state", {"expected": "torch.Tensor"}
        )
    expected = validation.state_names
    if state is not None and (
        not isinstance(state, StateTuple) or state.names != expected
    ):
        raise StatefulCompilationError(
            "RUNTIME_STATE", "initial_state", {"expected": list(expected)}
        )
    states: dict[str, torch.Tensor] = {}
    for index, spec in enumerate(validation.states):
        shape = (
            runtime.sequence_count,
            role_sizes[spec.head_role],
            *(role_sizes[role] for role in spec.feature_roles),
        )
        if state is None:
            states[spec.name] = _default_state(runtime, spec, shape, default_states)
            continue
        tensor = state[index]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or tensor.device != runtime.device
            or not tensor.is_contiguous()
        ):
            raise StatefulCompilationError(
                "RUNTIME_STATE",
                f"initial_state.{spec.name}",
                {
                    "shape": shape,
                    "dtype": "torch.float32",
                    "device": str(runtime.device),
                },
            )
        states[spec.name] = tensor
    return states


def _default_state(runtime, spec, shape, cache):
    key = (runtime.device, spec.name, shape, spec.identity)
    cached = None if cache is None else cache.get(key)
    if cached is None:
        cached = (
            torch.zeros(shape, device=runtime.device, dtype=torch.float32)
            if spec.identity == 0.0
            else torch.full(
                shape, spec.identity, device=runtime.device, dtype=torch.float32
            )
        )
        if cache is not None:
            cache[key] = cached
    return cached
