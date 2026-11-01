"""Immutable, zero-copy symbolic bindings for Stateful Operator inputs.

Bindings retain the authored tensor layout and expose pointer/stride metadata to
fused consumers.  No head or token expansion is performed.  ``load`` is only a
reference adapter: raw BF16 is promoted to FP32 at use, exactly as a kernel
load should be.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import torch


from ...stateful_operator import Batch, FeatureRole, HeadRole, Sequence as SequenceRole

_PACKED_TOKEN = object()


@dataclass(frozen=True)
class DenseRawBindingRecipe:
    """Fixed axis map for raw loads after public runtime validation.

    Holds no tensor or workload dimensions. Each bind reads the current tensor's
    strides/head extent; public shape/dtype/layout/alias checks remain mandatory.
    """

    name: str
    axes: tuple[int | None, int | None, int | None, int | None]

    def bind(self, tensor: torch.Tensor, heads: int):
        batch, head, sequence, feature = self.axes
        strides = tensor.stride()
        return (
            tensor,
            (
                strides[batch] if batch is not None else 0,
                strides[head] if head is not None else 0,
                strides[sequence] if sequence is not None else 0,
                strides[feature] if feature is not None else 0,
            ),
            heads // tensor.shape[head] if head is not None else heads,
        )


def compile_dense_binding(spec, feature):
    """Resolve role positions once; dynamic extents and storage stay unbound."""
    roles = spec.roles
    head = next((role for role in roles if isinstance(role, HeadRole)), None)
    return DenseRawBindingRecipe(
        spec.name,
        tuple(
            roles.index(role) if role in roles else None
            for role in (Batch, head, SequenceRole, feature)
        ),
    )


@dataclass(frozen=True)
class BindingDescriptor:
    """Metadata for one authored input, indexed by logical token/head/features."""

    name: str
    tensor: torch.Tensor
    spec_roles: tuple[Any, ...]
    source_roles: tuple[Any, ...]
    feature_roles: tuple[FeatureRole, ...]
    source_shape: tuple[int, ...]
    source_strides: tuple[int, ...]
    storage_offset: int
    mode: str
    sequence_count: int
    token_count: int
    sequence_length: int | None
    source_heads: int
    target_heads: int
    head_group_size: int
    broadcast_axes: tuple[Any, ...]
    physical_strides: tuple[int, ...]
    sequence_offsets: torch.Tensor | tuple[int, ...] | None = None

    @property
    def raw(self) -> torch.Tensor:
        return self.tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def is_broadcast_head(self) -> bool:
        return self.source_heads == 1 and self.target_heads > 1

    def sequence_for_token(self, token: int) -> int:
        if token < 0 or token >= self.token_count:
            raise IndexError(f"token {token} outside [0, {self.token_count})")
        if self.mode == "dense":
            assert self.sequence_length is not None
            return token // self.sequence_length
        offsets = self.sequence_offsets
        if offsets is None:
            raise ValueError("packed binding requires sequence offsets")
        if isinstance(offsets, torch.Tensor):
            offsets = tuple(int(x) for x in offsets.detach().cpu().tolist())
        return min(max(bisect_right(offsets, token) - 1, 0), self.sequence_count - 1)

    def token_in_sequence(self, token: int) -> int:
        sequence = self.sequence_for_token(token)
        if self.mode == "dense":
            assert self.sequence_length is not None
            return token % self.sequence_length
        offsets = self.sequence_offsets
        assert offsets is not None
        return token - int(
            offsets[sequence].item()
            if isinstance(offsets, torch.Tensor)
            else offsets[sequence]
        )

    def _indices(
        self, token: int, target_head: int, features: tuple[int, ...]
    ) -> tuple[int, ...]:
        if token < 0 or token >= self.token_count:
            raise IndexError(f"token {token} outside [0, {self.token_count})")
        if target_head < 0 or target_head >= self.target_heads:
            raise IndexError(
                f"target head {target_head} outside [0, {self.target_heads})"
            )
        if len(features) != len(self.feature_roles):
            raise ValueError(
                f"expected {len(self.feature_roles)} feature indices, got {len(features)}"
            )
        sequence = self.sequence_for_token(token)
        local = self.token_in_sequence(token)
        feature_values = dict(zip(self.feature_roles, features, strict=True))
        indices: list[int] = []
        for role, extent in zip(self.source_roles, self.source_shape, strict=True):
            if role is _PACKED_TOKEN:
                index = token
            elif role is Batch:
                index = sequence if extent != 1 else 0
            elif role is SequenceRole:
                index = local if extent != 1 else 0
            elif isinstance(role, HeadRole):
                index = target_head // self.head_group_size
            elif isinstance(role, FeatureRole):
                index = feature_values.get(role, 0)
            else:
                index = 0
            if index < 0 or index >= extent:
                raise IndexError(f"index {index} outside source axis of size {extent}")
            indices.append(index)
        return tuple(indices)

    def offset(self, token: int, target_head: int = 0, *features: int) -> int:
        indices = self._indices(token, target_head, tuple(features))
        return self.storage_offset + sum(
            i * stride for i, stride in zip(indices, self.source_strides)
        )

    def load(self, token: int, target_head: int = 0, *features: int) -> torch.Tensor:
        """Load one raw element and promote it to FP32 at point of use."""
        return self.tensor[self._indices(token, target_head, tuple(features))].float()

    def materialize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Explicit opt-in contiguous materialization for consumers requiring it."""
        feature_shape = tuple(
            _feature_extent(
                role, self.feature_roles, self.source_roles, self.source_shape
            )
            for role in self.feature_roles
        )
        output = torch.empty(
            (self.token_count, self.target_heads, *feature_shape),
            device=self.tensor.device,
            dtype=dtype,
        )
        for token, head, *feature in product(
            range(self.token_count),
            range(self.target_heads),
            *[range(n) for n in feature_shape],
        ):
            output[(token, head, *feature)] = self.load(token, head, *feature).to(dtype)
        return output


@dataclass(frozen=True)
class BindingSet:
    bindings: tuple[BindingDescriptor, ...]

    def __getitem__(self, name: str) -> BindingDescriptor:
        for binding in self.bindings:
            if binding.name == name:
                return binding
        raise KeyError(name)

    def __iter__(self):
        return iter(self.bindings)

    def as_dict(self) -> dict[str, BindingDescriptor]:
        return {binding.name: binding for binding in self.bindings}


def build_binding(
    spec: Any,
    tensor: torch.Tensor,
    *,
    role_sizes: Mapping[Any, int],
    mode: str = "dense",
    sequence_count: int | None = None,
    token_count: int | None = None,
    uniform_length: int | None = None,
    state_heads: int | None = None,
    sequence_offsets: torch.Tensor | Sequence[int] | None = None,
    feature_roles: Sequence[FeatureRole] | None = None,
) -> BindingDescriptor:
    """Build a descriptor from a raw tensor and ``TensorInput`` role spec."""
    if mode not in ("dense", "packed"):
        raise ValueError(f"unknown binding mode {mode!r}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"binding {getattr(spec, 'name', '<unnamed>')} is not a tensor")
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"binding tensors must be BF16 or FP32, got {tensor.dtype}")
    roles = tuple(spec.roles)
    if mode == "dense":
        source_roles = roles
        if SequenceRole in roles:
            length = int(tensor.shape[roles.index(SequenceRole)])
            sequences = int(tensor.shape[roles.index(Batch)]) if Batch in roles else 1
            tokens = sequences * length
        else:
            if uniform_length is None:
                raise ValueError(
                    "uniform_length is required for dense broadcast inputs"
                )
            sequences = int(
                sequence_count
                if sequence_count is not None
                else (tensor.shape[roles.index(Batch)] if Batch in roles else 1)
            )
            length, tokens = int(uniform_length), sequences * int(uniform_length)
        if sequence_count is not None and int(sequence_count) != sequences:
            raise ValueError("sequence_count disagrees with dense tensor shape")
        if token_count is not None and int(token_count) != tokens:
            raise ValueError("token_count disagrees with dense tensor shape")
        sequence_count, token_count, uniform_length = sequences, tokens, length
    else:
        if sequence_offsets is None:
            raise ValueError("packed bindings require sequence_offsets")
        if isinstance(sequence_offsets, torch.Tensor):
            if sequence_offsets.ndim != 1 or sequence_offsets.numel() < 2:
                raise ValueError(
                    "sequence_offsets must be a one-dimensional B+1 tensor"
                )
            # Keep device-local runtime offsets in the descriptor.  Do not copy
            # packed metadata to CPU: CUDA lowerings consume this pointer directly.
            if sequence_offsets.dtype not in (torch.int32, torch.int64):
                raise TypeError("sequence_offsets must be int32 or int64")
            sequences = int(sequence_offsets.numel() - 1)
            tokens = int(sequence_offsets[-1].item())
            if int(sequence_offsets[0].item()) != 0:
                raise ValueError("sequence_offsets must start at zero")
        else:
            offset_values = tuple(int(x) for x in sequence_offsets)
            if (
                len(offset_values) < 2
                or offset_values[0] != 0
                or any(a > b for a, b in zip(offset_values, offset_values[1:]))
            ):
                raise ValueError("sequence_offsets must be monotone and start at zero")
            sequences, tokens = len(offset_values) - 1, offset_values[-1]
        if tensor.ndim != len(
            tuple(role for role in roles if role not in (Batch, SequenceRole))
        ) + 1 or (tensor.ndim and tensor.shape[0] != tokens):
            raise ValueError(
                f"packed tensor {spec.name} must have token-major first axis of length {tokens}"
            )
        if sequence_count is not None and int(sequence_count) != sequences:
            raise ValueError("sequence_count disagrees with sequence_offsets")
        if token_count is not None and int(token_count) != tokens:
            raise ValueError("token_count disagrees with sequence_offsets")
        source_roles = (
            _PACKED_TOKEN,
            *(role for role in roles if role not in (Batch, SequenceRole)),
        )
        sequence_count, token_count, uniform_length = sequences, tokens, None
    if len(source_roles) != tensor.ndim:
        raise ValueError(
            f"{spec.name} roles describe {len(source_roles)} axes, tensor has {tensor.ndim}"
        )
    for role, actual in zip(source_roles, tensor.shape):
        if (
            role in role_sizes
            and role not in (Batch, SequenceRole)
            and not isinstance(role, HeadRole)
            and int(actual) != int(role_sizes[role])
        ):
            raise ValueError(
                f"{spec.name} axis {role} has {actual}, expected {role_sizes[role]}"
            )
    head_role = next(
        (role for role in source_roles if isinstance(role, HeadRole)), None
    )
    source_heads = (
        int(tensor.shape[source_roles.index(head_role)]) if head_role is not None else 1
    )
    target_heads = int(
        state_heads
        if state_heads is not None
        else role_sizes.get(head_role, source_heads)
    )
    if source_heads < 1 or target_heads < 1 or target_heads % source_heads:
        raise ValueError(
            f"target heads ({target_heads}) must be divisible by source heads ({source_heads})"
        )
    all_features = tuple(
        feature_roles or tuple(role for role in roles if isinstance(role, FeatureRole))
    )
    broadcasts = tuple(role for role in all_features if role not in source_roles)
    if head_role is None and target_heads > 1:
        broadcasts = (*broadcasts, "head")
    return BindingDescriptor(
        name=str(spec.name),
        tensor=tensor,
        spec_roles=roles,
        source_roles=source_roles,
        feature_roles=all_features,
        source_shape=tuple(tensor.shape),
        source_strides=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        mode=mode,
        sequence_count=int(sequence_count),
        token_count=int(token_count),
        sequence_length=uniform_length,
        source_heads=source_heads,
        target_heads=target_heads,
        head_group_size=target_heads // source_heads,
        broadcast_axes=broadcasts,
        physical_strides=tuple(tensor.stride()),
        sequence_offsets=sequence_offsets if mode == "packed" else None,
    )


def build_bindings(
    algorithm: Any,
    inputs: Mapping[str, torch.Tensor],
    *,
    role_sizes: Mapping[Any, int],
    mode: str = "dense",
    sequence_count: int | None = None,
    token_count: int | None = None,
    uniform_length: int | None = None,
    state_heads: int | None = None,
    sequence_offsets: torch.Tensor | Sequence[int] | None = None,
) -> BindingSet:
    """Build descriptors for all authored inputs in declaration order."""
    features: list[FeatureRole] = []
    for spec in (*algorithm.inputs, *algorithm.states):
        for role in spec.roles:
            if isinstance(role, FeatureRole) and role not in features:
                features.append(role)
    return BindingSet(
        tuple(
            build_binding(
                spec,
                inputs[spec.name],
                role_sizes=role_sizes,
                mode=mode,
                sequence_count=sequence_count,
                token_count=token_count,
                uniform_length=uniform_length,
                state_heads=state_heads,
                sequence_offsets=sequence_offsets,
                feature_roles=features,
            )
            for spec in algorithm.inputs
        )
    )


def _feature_extent(
    role: FeatureRole,
    all_roles: tuple[FeatureRole, ...],
    source_roles: tuple[Any, ...],
    shape: tuple[int, ...],
) -> int:
    return int(shape[source_roles.index(role)]) if role in source_roles else 1
