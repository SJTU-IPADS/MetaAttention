from dataclasses import asdict, dataclass, field
from enum import Enum
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Mapping

import torch

from ..runtime.workspace import WorkspacePolicy as WorkspacePolicy
from ..runtime.workspace import FACTORED_INFERENCE_WINDOW as FACTORED_INFERENCE_WINDOW
from .device import (
    DEFAULT_DEVICE_PROFILE,
    DeviceProfile,
    resolve_device_profile,
    schedule_configurations,
)


class PlanRepresentation(str, Enum):
    GENERIC_DENSE_AFFINE = "generic-dense-affine"
    VECTOR_PROPAGATION = "vector-propagation"
    AXIS_SCALE = "axis-scale"
    PRODUCT_INJECTION_GEMM = "product-injection-gemm"
    RANK_ONE_FACTORIZED = "rank-one-factorized"
    SHARED_STATE_CONTRACTION = "shared-state-contraction"
    SCALAR_FACTORIZED = "scalar-factorized"
    FACTORED_RECURRENCE = "factored-recurrence"
    RANK_ONE_DIAGONAL = "rank-one-diagonal"
    NATIVE_NORMALIZED = "native-normalized"
    NATIVE_DIAGONAL = "native-diagonal"
    PRODUCT_FEATURE_CHUNKED = "product-feature-chunked"
    CHUNKED_STATEFUL = "chunked-stateful"
    STATEFUL_REFERENCE = "stateful-reference"


@dataclass(frozen=True)
class MathematicalPlan:
    identity: str
    representation: PlanRepresentation
    rewrites: tuple[str, ...]
    semantic_regions: tuple[str, ...]
    preconditions: tuple[str, ...] = ()
    vjp_equivalence: str = "compiler-derived-vjp"
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanRule:
    representation: PlanRepresentation
    preconditions: tuple[str, ...]
    conflicts: tuple[str, ...] = ()


PLAN_RULES = {
    PlanRepresentation.GENERIC_DENSE_AFFINE: PlanRule(
        PlanRepresentation.GENERIC_DENSE_AFFINE,
        ("validated-scanable-affine-program",),
    ),
    PlanRepresentation.VECTOR_PROPAGATION: PlanRule(
        PlanRepresentation.VECTOR_PROPAGATION,
        (
            "typed-feature-vector-propagation",
            "rank-one-injection",
            "first-feature-readout",
        ),
        (PlanRepresentation.RANK_ONE_FACTORIZED.value,),
    ),
    PlanRepresentation.AXIS_SCALE: PlanRule(
        PlanRepresentation.AXIS_SCALE,
        ("typed-axis-scale-propagation", "authored-propagation-order"),
        (PlanRepresentation.RANK_ONE_FACTORIZED.value,),
    ),
    PlanRepresentation.PRODUCT_INJECTION_GEMM: PlanRule(
        PlanRepresentation.PRODUCT_INJECTION_GEMM,
        ("typed-role-compatible-product-injection",),
    ),
    PlanRepresentation.RANK_ONE_FACTORIZED: PlanRule(
        PlanRepresentation.RANK_ONE_FACTORIZED,
        ("typed-ordered-rank-one-propagation",),
    ),
    PlanRepresentation.SHARED_STATE_CONTRACTION: PlanRule(
        PlanRepresentation.SHARED_STATE_CONTRACTION,
        ("multiple-independent-readouts-of-shared-state",),
    ),
    PlanRepresentation.SCALAR_FACTORIZED: PlanRule(
        PlanRepresentation.SCALAR_FACTORIZED,
        ("scalar-propagation", "two-disjoint-single-feature-factors", "one-readout"),
        (
            PlanRepresentation.RANK_ONE_FACTORIZED.value,
            PlanRepresentation.SHARED_STATE_CONTRACTION.value,
        ),
    ),
    PlanRepresentation.FACTORED_RECURRENCE: PlanRule(
        PlanRepresentation.FACTORED_RECURRENCE,
        (
            "contracting-first-feature-readout",
            "disjoint-single-feature-injections",
            "ordered-axis-scale-and-rank-one-propagations",
        ),
    ),
    PlanRepresentation.PRODUCT_FEATURE_CHUNKED: PlanRule(
        PlanRepresentation.PRODUCT_FEATURE_CHUNKED,
        (
            "typed-product-feature-injections",
            "chunk-boundary-state-carries",
            "typed-readout-expressions",
        ),
    ),
    PlanRepresentation.CHUNKED_STATEFUL: PlanRule(
        PlanRepresentation.CHUNKED_STATEFUL,
        (
            "typed-stateful-injections",
            "chunk-boundary-state-carries",
            "typed-readout-expressions",
        ),
    ),
    PlanRepresentation.RANK_ONE_DIAGONAL: PlanRule(
        PlanRepresentation.RANK_ONE_DIAGONAL,
        (
            "typed-rank-one-state",
            "diagonal-affine-transition",
            "identity-state-readout",
        ),
    ),
    PlanRepresentation.NATIVE_NORMALIZED: PlanRule(
        PlanRepresentation.NATIVE_NORMALIZED,
        (
            "typed-normalized-readout",
            "native-vector-bindings",
            "chunk-boundary-state-carries",
        ),
    ),
    PlanRepresentation.NATIVE_DIAGONAL: PlanRule(
        PlanRepresentation.NATIVE_DIAGONAL,
        (
            "typed-diagonal-transition",
            "native-expression-bindings",
            "chunk-boundary-state-carries",
        ),
    ),
    PlanRepresentation.STATEFUL_REFERENCE: PlanRule(
        PlanRepresentation.STATEFUL_REFERENCE,
        (
            "input-driven-affine-state-transitions",
            "prior-and-current-state-readouts",
            "authored-step-order-and-acyclic-current-state-dependencies",
        ),
    ),
}


@dataclass(frozen=True)
class LoweredCandidate:
    identity: str
    plan_identity: str
    lowerer: str
    direction: str


@dataclass(frozen=True)
class KernelConfig:
    identity: str
    values: tuple[tuple[str, int], ...]


# A moved byte is the unit of the static estimate. On this class of device a
# floating-point operation costs a fraction of the byte that feeds it, and one
# launch costs a fixed slab of traffic it could otherwise have overlapped.
_FLOPS_PER_BYTE = 1 / 16
_BYTES_PER_LAUNCH = 4_096

# The representations whose runtime composes a per-token summary, runs one prefix
# pass over it, and contracts the readout in a separate pass.
_SUMMARY_REPRESENTATIONS = frozenset(
    {
        PlanRepresentation.AXIS_SCALE,
        PlanRepresentation.PRODUCT_INJECTION_GEMM,
        PlanRepresentation.RANK_ONE_FACTORIZED,
        PlanRepresentation.SHARED_STATE_CONTRACTION,
    }
)


@dataclass(frozen=True)
class EstimationShape:
    """The feature sizes and live program structure priced for a candidate."""

    heads: int
    rows: int
    columns: int
    dense_summary: bool
    state_count: int = 1
    state_total_elements: int = 0
    dense_total_elements: int = 0
    live_state_elements: int = 0
    backward: bool = False
    chunked: bool = False
    chunked_states: bool = False
    chunked_states_workspace: int = 0
    chunked_states_chunk_bytes: int = 0
    factored_workspace: int = 0
    factored_chunk_bytes: int = 0
    native_resources: Any | None = None
    joint_axis_size: int = 0

    @classmethod
    def of(
        cls,
        heads: int,
        feature_sizes: tuple[int, ...],
        dense_summary: bool,
        *,
        state_count: int = 1,
        state_total_elements: int = 0,
        dense_total_elements: int = 0,
        live_state_elements: int = 0,
        backward: bool = False,
        chunked: bool = False,
        chunked_states: bool = False,
        chunked_states_workspace: int = 0,
        chunked_states_chunk_bytes: int = 0,
        factored_workspace: int = 0,
        factored_chunk_bytes: int = 0,
        native_resources: Any | None = None,
        joint_axis_size: int = 0,
    ) -> "EstimationShape":
        rows = feature_sizes[0]
        columns = feature_sizes[1] if len(feature_sizes) > 1 else 1
        total = state_total_elements or state_count * rows * columns
        return cls(
            heads,
            rows,
            columns,
            dense_summary,
            state_count,
            total,
            dense_total_elements or total,
            live_state_elements or total,
            backward,
            chunked,
            chunked_states,
            chunked_states_workspace,
            chunked_states_chunk_bytes,
            factored_workspace,
            factored_chunk_bytes,
            native_resources,
            joint_axis_size,
        )

    @property
    def state_elements(self) -> int:
        return self.rows * self.columns

    @property
    def summary_elements(self) -> int:
        if self.dense_summary:
            return (
                self.rows * self.rows
                + self.columns * self.columns
                + self.state_elements
            )
        return self.state_elements


@dataclass(frozen=True)
class CandidateEstimate:
    """Static cost components of one Lowered Candidate for one shape.

    Every component is derived from the representation's own execution shape and
    the concrete Feature sizes, so the same representation is priced differently
    for different Shapes instead of carrying one constant.
    """

    feasible: bool
    bytes_per_token: int
    flops_per_token: int
    launches: int
    workspace_bytes_per_token: int
    workspace_bytes_per_chunk: int = 0
    aligned_workspace_bytes_per_token: int | None = None
    alignment_tokens: int = 0
    reason: str | None = None

    @property
    def cost(self) -> float:
        return (
            self.bytes_per_token
            + self.flops_per_token * _FLOPS_PER_BYTE
            + self.launches * _BYTES_PER_LAUNCH
        )


@dataclass(frozen=True)
class CandidateLowerer:
    name: str
    representations: tuple[PlanRepresentation, ...]
    runtime_family: str = "generic"

    def lower(
        self,
        plan: MathematicalPlan,
        shape: EstimationShape,
        *,
        maximum_workspace_bytes: int | None = None,
    ) -> LoweredCandidate | None:
        try:
            representation = PlanRepresentation(plan.representation)
        except ValueError:
            return None
        if representation not in self.representations:
            return None
        estimate = self.estimate(plan, shape)
        if not estimate.feasible:
            return None
        if (
            maximum_workspace_bytes is not None
            and estimate.workspace_bytes_per_token
            + 2 * estimate.workspace_bytes_per_chunk
            > maximum_workspace_bytes
        ):
            return None
        return LoweredCandidate(
            _digest((plan.identity, self.name, "forward")),
            plan.identity,
            self.name,
            "forward",
        )

    def estimate(
        self, plan: MathematicalPlan, shape: EstimationShape
    ) -> CandidateEstimate:
        """Price one representation for one Shape.

        The summary families move their per-token summary, the State they write,
        and the output. The factorized families move only their factors and the
        output. The dense and reference families also move the summary matrices
        and the union layout they materialize. The workspace component is the
        per-token buffer the plan has to keep live, so the same number gates
        feasibility and becomes the published `WorkspacePolicy`.
        """
        representation = PlanRepresentation(plan.representation)
        heads = shape.heads
        state = shape.live_state_elements
        summary = shape.summary_elements
        dense = shape.dense_total_elements

        if representation is PlanRepresentation.VECTOR_PROPAGATION:
            return CandidateEstimate(
                True,
                heads * (shape.rows + 2 * shape.columns) * 4,
                3 * heads * state,
                1,
                0,
                workspace_bytes_per_chunk=3 * heads * state * 4,
            )
        if representation is PlanRepresentation.SCALAR_FACTORIZED:
            return CandidateEstimate(
                True,
                heads * (shape.rows + 2 * shape.columns + 1) * 4,
                3 * heads * state,
                3,
                0,
                workspace_bytes_per_chunk=heads
                * (2 * shape.rows * shape.columns + 1)
                * 4,
            )
        if representation is PlanRepresentation.FACTORED_RECURRENCE:
            # Inference keeps affine chunk summaries and carries; backward
            # retains the original recorded-State reverse walk.
            return CandidateEstimate(
                True,
                heads * (shape.rows + 2 * shape.columns) * 4,
                4 * heads * state,
                1 if shape.backward else 3,
                max(heads * state * 2, shape.factored_workspace)
                if shape.backward
                else shape.factored_workspace,
                workspace_bytes_per_chunk=0
                if shape.backward
                else shape.factored_chunk_bytes,
            )
        if representation is PlanRepresentation.RANK_ONE_DIAGONAL:
            return CandidateEstimate(
                True,
                heads * (3 * state + shape.columns) * 4,
                3 * heads * state,
                3,
                shape.chunked_states_workspace,
                workspace_bytes_per_chunk=shape.chunked_states_chunk_bytes,
            )
        if representation is PlanRepresentation.NATIVE_NORMALIZED:
            return CandidateEstimate(
                True,
                heads * (3 * state + shape.columns) * 4,
                3 * heads * state,
                3,
                shape.chunked_states_workspace,
                workspace_bytes_per_chunk=shape.chunked_states_chunk_bytes,
            )
        if representation is PlanRepresentation.NATIVE_DIAGONAL:
            if shape.native_resources is None:
                return CandidateEstimate(
                    False, 0, 0, 0, 0, reason="native-diagonal-resources-missing"
                )
            resources = shape.native_resources
            token_bytes, chunk_bytes = resources.output_bytes, resources.chunk_bytes
            launches = resources.launches
            if resources.fused_softmax and not self.name.endswith("readout-cut"):
                token_bytes += resources.replay_serial_bytes
            if self.name == "tilelang-native-diagonal-shared-scores":
                chunk_bytes += resources.score_bytes
                launches += resources.score_launches
            elif self.name == "tilelang-native-diagonal-readout-cut":
                token_bytes += resources.cut_bytes
                launches += resources.cut_launches
            elif resources.cut_bytes and not resources.fused_softmax:
                token_bytes += resources.cut_bytes
            return CandidateEstimate(
                True,
                token_bytes + chunk_bytes // 64,
                resources.flops_per_token,
                launches,
                resources.output_bytes
                + (
                    resources.cut_bytes
                    if self.name.endswith("readout-cut") or not resources.fused_softmax
                    else 0
                ),
                workspace_bytes_per_chunk=chunk_bytes,
            )
        if representation in _SUMMARY_REPRESENTATIONS:
            arithmetic = 2 * heads * state * (2 if shape.dense_summary else 1)
            if representation is PlanRepresentation.AXIS_SCALE and shape.chunked:
                # Chunk replay keeps boundary buffers for every batch/sequence.
                return CandidateEstimate(
                    True,
                    heads * (3 * state + shape.columns) * 4,
                    arithmetic,
                    3,
                    heads * (summary + 2 * state) * 4,
                    workspace_bytes_per_chunk=3 * heads * state * 4,
                )
            return CandidateEstimate(
                True,
                heads * (summary + 2 * state + shape.columns) * 4,
                arithmetic,
                3,
                heads * (summary + 2 * state) * 4,
            )
        if representation in {
            PlanRepresentation.PRODUCT_FEATURE_CHUNKED,
            PlanRepresentation.CHUNKED_STATEFUL,
        }:
            if shape.native_resources is not None:
                resources = shape.native_resources
                token_bytes, chunk_bytes = resources.output_bytes, resources.chunk_bytes
                launches = resources.launches
                if resources.fused_softmax and not self.name.endswith("readout-cut"):
                    token_bytes += resources.replay_serial_bytes
                if self.name == "tilelang-chunked-shared-scores":
                    chunk_bytes += resources.score_bytes
                    launches += resources.score_launches
                elif self.name == "tilelang-chunked-readout-cut":
                    token_bytes += resources.cut_bytes
                    launches += resources.cut_launches
                elif resources.cut_bytes and not resources.fused_softmax:
                    token_bytes += resources.cut_bytes
                return CandidateEstimate(
                    True,
                    token_bytes + chunk_bytes // 64,
                    resources.flops_per_token,
                    launches,
                    resources.output_bytes
                    + (
                        resources.cut_bytes
                        if self.name.endswith("readout-cut")
                        or not resources.fused_softmax
                        else 0
                    ),
                    workspace_bytes_per_chunk=chunk_bytes,
                )
            # The chunked schedule keeps boundary carries and contracted
            # outputs; it does not materialize a token-sized State history.
            return CandidateEstimate(
                True,
                heads * (3 * shape.state_total_elements + shape.columns) * 4,
                3 * heads * shape.state_total_elements,
                3,
                shape.chunked_states_workspace,
                workspace_bytes_per_chunk=shape.chunked_states_chunk_bytes,
            )
        if representation is PlanRepresentation.GENERIC_DENSE_AFFINE:
            return CandidateEstimate(
                True,
                heads * (2 * dense + 2 * state) * 4,
                2 * heads * state * (shape.rows + shape.columns),
                3,
                heads * (2 * dense + 2 * state) * 4,
            )
        if representation is PlanRepresentation.STATEFUL_REFERENCE:
            if shape.joint_axis_size:
                # Joint block maps, additive trajectories, prefix temporaries
                # and autograd context. No dense transform over vec(K*V).
                width = shape.joint_axis_size
                elements = width * width + shape.state_total_elements
                return CandidateEstimate(
                    True,
                    heads * 8 * elements * 4,
                    2 * heads * width * elements,
                    4,
                    heads * 8 * elements * 4,
                )
            return CandidateEstimate(
                True,
                heads * (2 * dense + 4 * shape.state_total_elements) * 4,
                4 * heads * shape.state_total_elements * (shape.rows + shape.columns),
                4,
                heads * (2 * dense + 4 * shape.state_total_elements) * 4,
            )
        # every State pays for its own features, so the charge sums per State
        # rather than multiplying the first State's size by their count
        states = shape.state_total_elements
        if not shape.dense_summary:
            # without a rank-one contraction the reference lowering keeps one
            # elementwise factor per axis, so it materializes the factor, the
            # injection, and the per-token State rather than dense matrices
            if shape.chunked_states:
                # The dense schedule supports tails and keeps boundary carries,
                # factors and contracted terms, never a token State history.
                return CandidateEstimate(
                    True,
                    heads * (3 * states + shape.columns) * 4,
                    3 * heads * states,
                    3 * shape.state_count,
                    shape.chunked_states_workspace,
                    workspace_bytes_per_chunk=shape.chunked_states_chunk_bytes,
                )
            return CandidateEstimate(
                True,
                heads * (3 * states + shape.columns) * 4,
                3 * heads * states,
                3,
                heads * 3 * states * 4,
            )
        return CandidateEstimate(
            True,
            heads * (2 * dense + 4 * states) * 4,
            4 * heads * states * (shape.rows + shape.columns),
            4,
            heads * (2 * dense + 4 * states) * 4,
        )


LOWERERS = (
    CandidateLowerer("tilelang-generic", (PlanRepresentation.GENERIC_DENSE_AFFINE,)),
    CandidateLowerer(
        "tilelang-vector-propagation",
        (PlanRepresentation.VECTOR_PROPAGATION,),
        "vector-propagation",
    ),
    CandidateLowerer(
        "tilelang-elementwise", (PlanRepresentation.AXIS_SCALE,), "axis-scale"
    ),
    CandidateLowerer(
        "tilelang-product-gemm",
        (PlanRepresentation.PRODUCT_INJECTION_GEMM,),
        "product-injection-gemm",
    ),
    CandidateLowerer(
        "tilelang-rank-one",
        (PlanRepresentation.RANK_ONE_FACTORIZED,),
        "rank-one-factorized",
    ),
    CandidateLowerer(
        "tilelang-shared-contraction",
        (PlanRepresentation.SHARED_STATE_CONTRACTION,),
        "shared-state-contraction",
    ),
    CandidateLowerer(
        "tilelang-scalar-factorized",
        (PlanRepresentation.SCALAR_FACTORIZED,),
        "scalar-factorized",
    ),
    CandidateLowerer(
        "tilelang-factored-recurrence",
        (PlanRepresentation.FACTORED_RECURRENCE,),
        "factored-recurrence",
    ),
    CandidateLowerer(
        "tilelang-product-feature-chunked",
        (PlanRepresentation.PRODUCT_FEATURE_CHUNKED,),
        "product-feature-chunked",
    ),
    CandidateLowerer(
        "tilelang-chunked-stateful",
        (PlanRepresentation.CHUNKED_STATEFUL,),
        "chunked-stateful",
    ),
    CandidateLowerer(
        "tilelang-chunked-shared-scores",
        (PlanRepresentation.CHUNKED_STATEFUL,),
        "chunked-stateful",
    ),
    CandidateLowerer(
        "tilelang-chunked-readout-cut",
        (PlanRepresentation.CHUNKED_STATEFUL,),
        "chunked-stateful",
    ),
    CandidateLowerer(
        "tilelang-rank-one-diagonal",
        (PlanRepresentation.RANK_ONE_DIAGONAL,),
        "rank-one-diagonal",
    ),
    CandidateLowerer(
        "tilelang-native-normalized",
        (PlanRepresentation.NATIVE_NORMALIZED,),
        "native-normalized",
    ),
    CandidateLowerer(
        "tilelang-native-diagonal",
        (PlanRepresentation.NATIVE_DIAGONAL,),
        "native-diagonal",
    ),
    CandidateLowerer(
        "tilelang-native-diagonal-shared-scores",
        (PlanRepresentation.NATIVE_DIAGONAL,),
        "native-diagonal",
    ),
    CandidateLowerer(
        "tilelang-native-diagonal-readout-cut",
        (PlanRepresentation.NATIVE_DIAGONAL,),
        "native-diagonal",
    ),
    CandidateLowerer(
        "reference-stateful",
        (PlanRepresentation.STATEFUL_REFERENCE,),
        "stateful-reference",
    ),
)


def _backward_configuration(
    forward_values: tuple[tuple[str, int], ...],
    family: str,
    profile: DeviceProfile,
) -> tuple[tuple[str, int], ...]:
    """Reuse the selected forward launch width when backward uses that knob."""
    name = "block_tokens" if family == "elementwise" else "threads"
    values = dict(forward_values)
    fallback = (
        profile.default_block_tokens[0]
        if name == "block_tokens"
        else profile.default_threads[0]
    )
    return ((name, int(values.get(name, fallback))),)


def backward_context_family(
    algorithm: Any, candidate: LoweredCandidate, dense_summary: bool
) -> str:
    """Select the backward context that matches the forward runtime branch.

    The forward pass picks its elementwise scan whenever the per-token summary is
    a scalar pair rather than dense ``(left, right)`` factors, so the generated
    backward must be bound to the same context or it would ask for values the
    forward never saved.
    """
    family = runtime_family_for(candidate)
    if family == "factored-recurrence":
        return "factored"
    if family == "axis-scale":
        return "dense" if dense_summary else "elementwise"
    if family in {"generic", "product-injection-gemm", "shared-state-contraction"}:
        # the forward composes a dense summary pair only when it was asked for
        # one, so the backward must read the elementwise context otherwise
        return "dense" if dense_summary else "elementwise"
    return "dense"


def runtime_family_for(candidate: LoweredCandidate) -> str:
    return next(
        lowerer.runtime_family
        for lowerer in LOWERERS
        if lowerer.name == candidate.lowerer
    )


def kernel_configurations(
    candidate: LoweredCandidate,
    profile: DeviceProfile | None = None,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Return lowering-owned schedule candidates for a target profile."""
    profile = DEFAULT_DEVICE_PROFILE if profile is None else profile
    return tuple(
        config.values for config in schedule_configurations(candidate, profile)
    )


def select_candidate(
    candidates: tuple[LoweredCandidate, ...],
    estimates: Mapping[Any, CandidateEstimate],
) -> LoweredCandidate:
    """Select by shape-aware estimate with a deterministic identity tie-break."""
    if not candidates:
        raise ValueError("no feasible lowered candidates")
    return min(
        candidates,
        key=lambda item: (estimates[item].cost, item.lowerer, item.identity),
    )


@dataclass(frozen=True)
class ContextValue:
    name: str
    dtype: str
    roles: tuple[str, ...]
    shape_rule: str


@dataclass(frozen=True)
class ContextContract:
    produced: tuple[ContextValue, ...]
    required: tuple[ContextValue, ...]
    recomputable: tuple[ContextValue, ...]

    @property
    def identity(self) -> str:
        return _digest(asdict(self))

    def validate(self) -> None:
        available = {
            value.name: value for value in (*self.produced, *self.recomputable)
        }
        if len(available) != len(self.produced) + len(self.recomputable):
            raise ValueError("duplicate context value name")
        for required in self.required:
            if available.get(required.name) != required:
                raise ValueError(f"incompatible context value: {required.name}")


@dataclass(frozen=True)
class Composition:
    forward_candidate: LoweredCandidate
    backward_candidate: LoweredCandidate | None
    context_contract: ContextContract

    def validate(self) -> None:
        self.context_contract.validate()
        if (
            self.backward_candidate is not None
            and self.backward_candidate.direction != "backward"
        ):
            raise ValueError(
                "composition backward candidate must lower backward direction"
            )


@dataclass(frozen=True)
class WorkloadPoint:
    sequence_count: int
    token_count: int
    distribution: str


@dataclass(frozen=True)
class WorkloadPolicy:
    version: str
    dense: tuple[WorkloadPoint, ...]
    packed: tuple[WorkloadPoint, ...]

    def points(self, mode: str) -> tuple[WorkloadPoint, ...]:
        return self.dense if mode == "dense" else self.packed


WORKLOAD_POLICY = WorkloadPolicy(
    "stateful-workloads/provisional-h20-v1",
    dense=(
        WorkloadPoint(1, 64, "uniform"),
        WorkloadPoint(2, 512, "uniform"),
        WorkloadPoint(4, 4096, "uniform"),
    ),
    packed=(
        WorkloadPoint(4, 256, "uniform"),
        WorkloadPoint(8, 1024, "long-tail"),
        WorkloadPoint(8, 512, "empty-containing"),
    ),
)


def normalized_geometric_mean(
    measurements: dict[str, tuple[float, ...]],
) -> dict[str, float]:
    import math

    if not measurements:
        return {}
    widths = {len(values) for values in measurements.values()}
    if len(widths) != 1 or not widths or next(iter(widths)) == 0:
        raise ValueError("candidate measurements must cover one common workload policy")
    width = next(iter(widths))
    best = tuple(
        min(values[index] for values in measurements.values()) for index in range(width)
    )
    return {
        candidate: math.exp(
            sum(math.log(value / best[index]) for index, value in enumerate(values))
            / width
        )
        for candidate, values in measurements.items()
    }


@dataclass(frozen=True)
class TileLangPlan:
    """Portable identity and environment descriptor for one specialization."""

    semantic_identity: str
    mathematical_plan: MathematicalPlan
    candidate: LoweredCandidate
    kernel_config: KernelConfig
    binary_identity: str
    tuning_identity: str
    mode: str
    feature_sizes: tuple[int, ...]
    context_contract: ContextContract
    workspace: WorkspacePolicy
    mathematical_plans: tuple[MathematicalPlan, ...] = ()
    lowered_candidates: tuple[LoweredCandidate, ...] = ()
    backward_candidate: LoweredCandidate | None = None
    backward_kernel_config: KernelConfig | None = None
    uses_parallel_summary: bool = True
    candidate_estimates: tuple[tuple[str, CandidateEstimate], ...] = ()
    estimation_shape: EstimationShape | None = None
    device_profile: DeviceProfile | None = None

    @property
    def backend(self) -> str:
        return "tilelang"
        return "tilelang"


@dataclass(frozen=True)
class TileLangExecutable:
    """Compiler-owned forward/backward dispatch for one analyzed specialization."""

    plan: TileLangPlan
    backward_ir: Any | None
    dense_summary: bool
    scalar_factorized: Any | None
    program: Any = field(repr=False, compare=False)
    rank_one_factored: Any | None = None
    symmetric_rank_one: bool = False
    composition: Composition | None = None
    backward_binding: Any | None = None
    chunked_recipe: Any | None = None
    _runtime_family: str = field(init=False, repr=False)
    _execution: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self, "_runtime_family", runtime_family_for(self.plan.candidate)
        )
        from ..runtime.execution import SelectedExecution

        object.__setattr__(self, "_execution", SelectedExecution(self.program, self))

    def execute(self, **validated_call):
        """Execute a validated invocation with the compiler-bound adapter."""
        return self._execution(**validated_call)

    @property
    def runtime_family(self) -> str:
        return self._runtime_family

    def _validate_gradient_dispatch(self, *tensors: torch.Tensor) -> None:
        if not torch.is_grad_enabled() or not any(
            tensor.requires_grad for tensor in tensors
        ):
            return
        if (
            self.backward_ir is None
            or self.backward_binding is None
            or self.plan.backward_candidate is None
            or self.composition is None
            or self.composition.backward_candidate != self.plan.backward_candidate
        ):
            raise RuntimeError(
                "gradient execution has no compiler-selected backward candidate"
            )

    def prefix(self, transform, bias, segment, initial_state, second=None):
        transform_inputs = transform if isinstance(transform, tuple) else (transform,)
        if second is not None:
            transform_inputs = (*transform_inputs, second)
        self._validate_gradient_dispatch(*transform_inputs, bias, initial_state)
        if isinstance(transform, tuple):
            from ..lowering.scan.affine import tilelang_dense_affine_prefix

            return tilelang_dense_affine_prefix(
                transform[0],
                transform[1],
                bias,
                segment,
                initial_state,
                backward=self.backward_binding,
                config=self.plan.kernel_config.values,
            )
        from ..lowering.scan.elementwise import tilelang_elementwise_prefix

        return tilelang_elementwise_prefix(
            transform,
            bias,
            segment,
            initial_state,
            backward=self.backward_binding,
            config=self.plan.kernel_config.values,
            second=second,
        )

    def vector_propagation_dense(self, **kwargs):
        """Run the feature-vector propagation kernel."""
        from ..lowering.scan.vector import vector_propagation_dense

        return vector_propagation_dense(**kwargs)

    def chunked_elementwise_dense(self, **kwargs):
        """Run the chunked diagonal schedule; the caller guarantees no gradient."""
        from ..lowering.chunked import chunked_elementwise_dense

        return chunked_elementwise_dense(**kwargs)

    def factored_recurrence_dense(self, **kwargs):
        from ..lowering.lowrank.factored import factored_recurrence_dense

        if kwargs.get("initial_state") is not None:
            self._validate_gradient_dispatch(
                kwargs["initial_state"], *kwargs.get("left_factors", ())
            )
        return factored_recurrence_dense(
            backward=self.backward_binding,
            config=self.plan.kernel_config.values,
            threads=dict(self.plan.kernel_config.values).get("threads", 128),
            **kwargs,
        )

    def scalar_factorized_dense_raw(self, raw_inputs, initial_state, **kwargs):
        from ..lowering.scalar.scalar import tilelang_scalar_factorized_dense_raw

        if self.scalar_factorized is None or self.scalar_factorized.dense is None:
            raise ValueError("plan has no raw dense scalar lowering")
        return tilelang_scalar_factorized_dense_raw(
            raw_inputs,
            initial_state,
            self.scalar_factorized.dense,
            threads=dict(self.plan.kernel_config.values).get("threads", 128),
            **kwargs,
        )

    def scalar_factorized_dense(self, *args, **kwargs):
        from ..lowering.scalar.scalar import tilelang_scalar_factorized_dense

        if self.scalar_factorized is None:
            raise ValueError("plan is not scalar-factorized")
        return tilelang_scalar_factorized_dense(
            *args,
            threads=dict(self.plan.kernel_config.values).get("threads", 128),
            **kwargs,
        )

    def scalar_factorized_packed(self, *args, **kwargs):
        from ..lowering.scalar.scalar import tilelang_scalar_factorized_forward

        if self.scalar_factorized is None:
            raise ValueError("plan is not scalar-factorized")
        return tilelang_scalar_factorized_forward(
            *args,
            threads=dict(self.plan.kernel_config.values).get("threads", 128),
            **kwargs,
        )


def publish_successful_artifact(identity: str, payload: Any, *, layer: str) -> Path:
    """Atomically publish one successful compiler artifact under its layer identity."""
    root = (
        Path(
            os.environ.get(
                "STATEFUL_OPERATOR_CACHE_DIR",
                Path.home() / ".cache" / "meta-attention" / "stateful-operator",
            )
        )
        / layer
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{identity}.json"
    lock_path = root / f"{identity}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return destination
        descriptor, temporary = tempfile.mkstemp(prefix=".artifact-", dir=root)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination


def compile_plan(
    *,
    source_identity: str,
    mode: str,
    feature_sizes: tuple[int, ...],
    target: Any,
    backward_ir: Any | None,
    dense_summary: bool,
    scalar_factorized: Any | None,
    rank_one_factored: Any | None,
    structural_features: Any,
    optimized: bool,
    autotune: bool,
    operator: Any,
    compilation: Any,
) -> TileLangExecutable:
    """Enumerate mathematical plans, then bind one stable executable candidate."""
    device_profile = resolve_device_profile(target)
    from ..lowering.chunked import compile_chunked_recipe

    chunked_recipe = (
        compile_chunked_recipe(operator._program, dict(compilation.role_sizes))
        if mode == "dense"
        else None
    )
    from ..lowering.structured import product_feature_chunked_eligibility

    product_feature_chunked = (
        optimized
        and mode == "dense"
        and backward_ir is None
        and target.device_type == "cuda"
        and product_feature_chunked_eligibility(operator._program)[0]
    )

    chunked_stateful = (
        optimized
        and mode == "dense"
        and backward_ir is None
        and target.device_type == "cuda"
        and chunked_recipe is not None
        and not product_feature_chunked
    )
    reference_required = operator._program._analysis.reference_required
    rewrites = _structured_rewrites(structural_features) if optimized else ()
    representations: list[PlanRepresentation] = []
    special_representation = None
    if (
        optimized
        and target.device_type == "cuda"
        and mode == "dense"
        and backward_ir is None
    ):
        if chunked_recipe is not None and chunked_recipe.rank_one is not None:
            special_representation = PlanRepresentation.RANK_ONE_DIAGONAL
        elif (
            chunked_recipe is not None and chunked_recipe.native_normalized is not None
        ):
            special_representation = PlanRepresentation.NATIVE_NORMALIZED
        elif chunked_recipe is not None and chunked_recipe.native_diagonal is not None:
            special_representation = PlanRepresentation.NATIVE_DIAGONAL
        elif product_feature_chunked:
            special_representation = PlanRepresentation.PRODUCT_FEATURE_CHUNKED
        elif chunked_recipe is not None:
            special_representation = PlanRepresentation.CHUNKED_STATEFUL
    if not reference_required:
        representations.append(PlanRepresentation.GENERIC_DENSE_AFFINE)
        representations.extend(rewrites)
        if optimized and scalar_factorized is not None:
            representations.append(PlanRepresentation.SCALAR_FACTORIZED)
    if (
        optimized
        and rank_one_factored is not None
        and (
            mode == "dense"
            or (
                target.device_type == "cuda"
                and (
                    rank_one_factored.transpose_state
                    or rank_one_factored.injection_scalar_count
                    or rank_one_factored.injection_propagations
                )
            )
        )
        and (not reference_required or target.device_type == "cuda")
    ):
        representations.append(PlanRepresentation.FACTORED_RECURRENCE)
    if special_representation is not None:
        representations.append(special_representation)
    if reference_required:
        representations.append(PlanRepresentation.STATEFUL_REFERENCE)
    if not representations:
        representations.append(PlanRepresentation.GENERIC_DENSE_AFFINE)
    representations = list(dict.fromkeys(representations))
    mathematical_plans = tuple(
        MathematicalPlan(
            _digest((source_identity, representation.value)),
            representation,
            tuple(item.value for item in rewrites)
            if representation is not PlanRepresentation.GENERIC_DENSE_AFFINE
            else (),
            ("state-transition", "state-readouts"),
            PLAN_RULES[representation].preconditions,
            "compiler-derived-vjp",
            PLAN_RULES[representation].conflicts,
        )
        for representation in representations
    )
    from ..lowering.structured import RUNTIME_FAMILY_ELIGIBILITY

    fixed_sizes = dict(compilation.role_sizes)
    state = operator._program.states[0]
    from ..lowering.chunked import chunked_states_workspace

    chunked_states = (
        mode == "dense"
        and backward_ir is None
        and target.device_type == "cuda"
        and chunked_recipe is not None
    )
    state_workspace, state_chunk_bytes = (
        chunked_states_workspace(
            operator._program, fixed_sizes, fixed_sizes[state.head_role]
        )
        if chunked_states
        else (0, 0)
    )
    if optimized and chunked_recipe is not None and chunked_recipe.rank_one is not None:
        elements = (
            fixed_sizes[state.head_role] * fixed_sizes[chunked_recipe.rank_one.feature]
        )
        # Three FP32 chunk vectors plus recursive groups; the extra chunk
        # conservatively covers final State.
        state_workspace, state_chunk_bytes = 4 * elements, 16 * elements
    if (
        optimized
        and chunked_recipe is not None
        and chunked_recipe.native_normalized is not None
    ):
        native = chunked_recipe.native_normalized
        memory = next(
            spec for spec in operator._program.states if spec.name == native.memory
        )
        heads = fixed_sizes[memory.head_role]
        rows, columns = (fixed_sizes[role] for role in memory.feature_roles)
        state_workspace = 4 * heads * columns
        state_chunk_bytes = 12 * heads * rows * (columns + 1)
    native_resources = None
    if (
        optimized
        and chunked_recipe is not None
        and chunked_recipe.native_product_normalized is not None
    ):
        from ..lowering.readout.product_native import native_product_estimate

        native_resources = native_product_estimate(
            chunked_recipe.native_product_normalized,
            fixed_sizes,
            fixed_sizes[state.head_role],
        )
        state_workspace = native_resources.output_bytes
        state_chunk_bytes = native_resources.chunk_bytes
    if chunked_recipe is not None and chunked_recipe.native_diagonal is not None:
        from ..lowering.diagonal.diagonal_native import native_diagonal_resources

        native_resources = native_diagonal_resources(
            chunked_recipe.native_diagonal,
            fixed_sizes,
            fixed_sizes[state.head_role],
        )
        state_workspace = native_resources.output_bytes
        state_chunk_bytes = native_resources.chunk_bytes
    factored_token_bytes = factored_chunk_bytes = 0
    if rank_one_factored is not None:
        heads = fixed_sizes[state.head_role]
        rows, columns = feature_sizes
        if rank_one_factored.transpose_state:
            rows, columns = columns, rows
        if backward_ir is not None:
            factored_token_bytes = (
                2 * heads * rows * columns
                + 4
                * heads
                * rank_one_factored.injection_scalar_count
                * (4 + (columns + 15) // 16)
            )
        else:
            operation_count = len(rank_one_factored.operations)
            injection_count = rank_one_factored.injection_count
            scalar_count = rank_one_factored.injection_scalar_count
            readout_count = len(rank_one_factored.readout_roles)
            query_width = columns if rank_one_factored.readout_axis else rows
            output_width = rows if rank_one_factored.readout_axis else columns
            factored_token_bytes = (
                4
                * heads
                * (
                    operation_count * (2 * rows + columns + 1)
                    + injection_count * (rows + columns)
                    + 5 * scalar_count
                    + readout_count * (query_width + 2 * output_width)
                    + 4
                    * (operation_count + injection_count + readout_count)
                    * (rows + columns)
                )
            )
            factored_chunk_bytes = (
                4 * heads * (rows * rows + 2 * rows * columns + columns)
            )
            from ..lowering.lowrank.wy_kernels import (
                eligible as wy_eligible,
                scratch_bytes_per_chunk,
            )

            operation_codes = tuple(
                operation.kind for operation in rank_one_factored.operations
            )
            if (
                mode == "dense"
                and not rank_one_factored.readout_axis
                and not scalar_count
                and wy_eligible(operation_codes, injection_count, rows, columns)
            ):
                factored_chunk_bytes = max(
                    factored_chunk_bytes,
                    heads
                    * scratch_bytes_per_chunk(
                        operation_codes,
                        injection_count,
                        readout_count,
                        rows,
                        columns,
                    ),
                )
        if rank_one_factored.injection_propagations:
            # Preparing ordered writes retains differentiable vector actions,
            # including their reverse intermediates, outside the kernel.
            suffix_actions = sum(map(len, rank_one_factored.injection_propagations))
            factored_token_bytes += 6 * 4 * heads * rows * suffix_actions
    state_total = 0
    for spec in operator._program.states:
        sizes = [fixed_sizes[role] for role in spec.feature_roles]
        state_total += sizes[0] if len(sizes) == 1 else sizes[0] * sizes[1]
    shape = EstimationShape.of(
        fixed_sizes[state.head_role],
        feature_sizes,
        dense_summary,
        state_count=len(operator._program.states),
        state_total_elements=state_total,
        joint_axis_size=(
            len(operator._program.states)
            * fixed_sizes[state.feature_roles[operator._program._analysis.joint_axis]]
            if operator._program._analysis.joint_axis is not None
            else 0
        ),
        dense_total_elements=sum(
            fixed_sizes[spec.feature_roles[0]] ** 2
            + (
                fixed_sizes[spec.feature_roles[1]] ** 2
                if len(spec.feature_roles) > 1
                else 0
            )
            + (state_total // max(1, len(operator._program.states)))
            for spec in operator._program.states
        ),
        chunked_states_workspace=state_workspace,
        chunked_states_chunk_bytes=(
            state_chunk_bytes
            if optimized and (chunked_stateful or product_feature_chunked)
            else 0
        ),
        factored_workspace=factored_token_bytes,
        factored_chunk_bytes=factored_chunk_bytes,
        native_resources=native_resources,
    )
    candidate_list: list[LoweredCandidate] = []
    estimates: dict[LoweredCandidate, CandidateEstimate] = {}
    for item in mathematical_plans:
        native_diagonal = (
            None if chunked_recipe is None else chunked_recipe.native_diagonal
        )
        for lowerer in LOWERERS:
            if lowerer.name in (
                "tilelang-chunked-shared-scores",
                "tilelang-native-diagonal-shared-scores",
            ) and (
                native_diagonal is None
                or native_diagonal.softmax is not None
                or any(len(state.injections) != 1 for state in native_diagonal.states)
                or not any(state.identity_columns for state in native_diagonal.states)
            ):
                continue
            if lowerer.name in (
                "tilelang-chunked-readout-cut",
                "tilelang-native-diagonal-readout-cut",
            ) and (
                native_diagonal is None
                or native_diagonal.softmax is None
                or not native_diagonal.fused_softmax
            ):
                continue
            if lowerer.runtime_family == "vector-propagation" and (
                mode != "dense"
                or backward_ir is not None
                or target.device_type != "cuda"
            ):
                continue
            if lowerer.runtime_family == "scalar-factorized" and (
                backward_ir is not None
                or scalar_factorized is None
                or scalar_factorized.dense is None
                or scalar_factorized.injection_count != 1
                or scalar_factorized.readout_count != 1
            ):
                # the plan is only its fused kernel: a compiled backward has no
                # binding for it, and anything else falls through to the summary
                # path this candidate would mis-price
                continue
            if (
                lowerer.runtime_family == "factored-recurrence"
                and backward_ir is not None
                and (
                    rank_one_factored is None
                    or any(
                        operation.kind
                        not in {"column-scale", "row-scale", "scale", "rank-one"}
                        for operation in rank_one_factored.operations
                    )
                )
            ):
                continue
            if (
                item.representation
                not in {
                    PlanRepresentation.GENERIC_DENSE_AFFINE,
                    PlanRepresentation.RANK_ONE_DIAGONAL,
                    PlanRepresentation.NATIVE_NORMALIZED,
                    PlanRepresentation.NATIVE_DIAGONAL,
                }
                and not RUNTIME_FAMILY_ELIGIBILITY.get(
                    lowerer.runtime_family, lambda _: (True, None)
                )(operator._program)[0]
            ):
                continue
            candidate = lowerer.lower(
                item, shape, maximum_workspace_bytes=target.workspace_bytes
            )
            if candidate is None:
                continue
            candidate_list.append(candidate)
            estimates[candidate] = lowerer.estimate(item, shape)
    candidates = tuple(candidate_list)
    if not candidates:
        from ...stateful_operator import StatefulCompilationError

        raise StatefulCompilationError(
            "COMPILE_RESOURCE",
            "target.workspace_bytes",
            {"reason": "no feasible lowered candidate"},
        )
    from .binding import CandidateBinding

    binding = CandidateBinding(
        program=operator._program,
        compilation=compilation,
        source_identity=source_identity,
        feature_sizes=feature_sizes,
        mathematical_plans=mathematical_plans,
        candidates=candidates,
        estimates=estimates,
        shape=shape,
        device_profile=device_profile,
        policy_version=WORKLOAD_POLICY.version,
        backward_ir=backward_ir,
        dense_summary=dense_summary,
        scalar_factorized=scalar_factorized,
        rank_one_factored=rank_one_factored,
        chunked_recipe=chunked_recipe,
    )
    candidate = select_candidate(candidates, estimates)
    values = kernel_configurations(candidate, device_profile)[0]
    if autotune:
        from .benchmark import invocation_factory
        from .tuning import MeasurementCache, tune_candidates

        device = torch.device(target.device_type)
        cache = None
        cache_root = os.environ.get("STATEFUL_OPERATOR_CACHE_DIR")
        if cache_root:
            cache = MeasurementCache(
                str(Path(cache_root) / "tuning" / "measurements.json")
            )
        factory = invocation_factory(operator, compilation, binding, device)
        tuning_result = tune_candidates(
            candidates,
            {item: kernel_configurations(item, device_profile) for item in candidates},
            WORKLOAD_POLICY,
            factory,
            autotune=True,
            mode=mode,
            environment=(device_profile.target_identity, _target_fingerprint(target)),
            cache=cache,
            static_costs={item: int(estimates[item].cost) for item in candidates},
        )
        candidate = tuning_result.winner
        values = tuning_result.winner_config
    executable = binding.bind(candidate, values)
    plan = executable.plan
    if os.environ.get("STATEFUL_OPERATOR_CACHE_DIR"):
        publish_successful_artifact(
            plan.mathematical_plan.identity,
            asdict(plan.mathematical_plan),
            layer="plans",
        )
        publish_successful_artifact(
            plan.kernel_config.identity, asdict(plan.kernel_config), layer="configs"
        )
        publish_successful_artifact(
            plan.binary_identity, asdict(plan), layer="binaries"
        )
        publish_successful_artifact(
            plan.context_contract.identity,
            asdict(plan.context_contract),
            layer="contexts",
        )
        if plan.backward_candidate is not None:
            publish_successful_artifact(
                plan.backward_candidate.identity,
                asdict(plan.backward_candidate),
                layer="backward",
            )
    return executable


def _structured_rewrites(features: Any) -> tuple[PlanRepresentation, ...]:
    rewrites = []
    if features.vector_propagation:
        rewrites.append(PlanRepresentation.VECTOR_PROPAGATION)
    if features.axis_scale:
        rewrites.append(PlanRepresentation.AXIS_SCALE)
    if features.product_injection:
        rewrites.append(PlanRepresentation.PRODUCT_INJECTION_GEMM)
    if features.rank_one:
        rewrites.append(PlanRepresentation.RANK_ONE_FACTORIZED)
    if features.shared_state_contraction:
        rewrites.append(PlanRepresentation.SHARED_STATE_CONTRACTION)
    return tuple(rewrites)


def _target_fingerprint(target: Any) -> tuple[Any, ...]:
    return (
        target.device_type,
        target.product_family,
        target.capability,
        target.workspace_bytes,
    )


def _toolchain_fingerprint() -> tuple[str, ...]:
    try:
        tilelang_version = version("tilelang")
    except PackageNotFoundError:
        tilelang_version = "unavailable"
    return (
        platform.python_implementation(),
        platform.python_version(),
        torch.__version__,
        str(torch.version.cuda),
        tilelang_version,
        "stateful-lowering/tilelang-1",
    )


def _environment_fingerprint(device: torch.device) -> tuple[Any, ...]:
    if device.type != "cuda":
        return (device.type,)
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return (
        properties.name,
        tuple(torch.cuda.get_device_capability(index)),
        properties.total_memory,
        properties.multi_processor_count,
        _nvidia_smi_fingerprint(index, "driver_version"),
        _toolchain_fingerprint(),
    )


def _clock_fingerprint(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _nvidia_smi_fingerprint(
        index,
        "clocks.applications.graphics,clocks.applications.memory,persistence_mode,power.limit",
    )


def _nvidia_smi_fingerprint(index: int, fields: str) -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                "--id=" + str(torch.cuda.get_device_properties(index).uuid),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "unable to fingerprint the CUDA driver and clock policy"
        ) from error


@dataclass(frozen=True)
class EvidenceStatus:
    target: str
    workload_policy: str
    status: str
    note: str


H20_EVIDENCE = EvidenceStatus(
    "NVIDIA H20",
    WORKLOAD_POLICY.version,
    "pending",
    "Freeze policy and record measurements on an otherwise idle H20.",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def is_parallel_plan(plan: Any) -> bool:
    return isinstance(plan, TileLangExecutable) and plan.plan.uses_parallel_summary
