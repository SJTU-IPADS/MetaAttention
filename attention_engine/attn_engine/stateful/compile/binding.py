"""Bind a lowered candidate identically for tuning and published execution.

The binding retains the unselected recipes. Every bind derives a fresh, complete
executable, so probing one schedule cannot constrain a later candidate.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from .device import DeviceProfile
from .plan import (
    Composition,
    ContextContract,
    ContextValue,
    EstimationShape,
    CandidateEstimate,
    KernelConfig,
    LoweredCandidate,
    MathematicalPlan,
    TileLangExecutable,
    TileLangPlan,
    _backward_configuration,
    _digest,
    _target_fingerprint,
    _toolchain_fingerprint,
    backward_context_family,
    kernel_configurations,
)
from ..runtime.workspace import WorkspacePolicy


@dataclass(frozen=True)
class CandidateBinding:
    program: Any
    compilation: Any
    source_identity: str
    feature_sizes: tuple[int, ...]
    mathematical_plans: tuple[MathematicalPlan, ...]
    candidates: tuple[LoweredCandidate, ...]
    estimates: Mapping[LoweredCandidate, CandidateEstimate]
    shape: EstimationShape
    device_profile: DeviceProfile
    policy_version: str
    backward_ir: Any | None
    dense_summary: bool
    scalar_factorized: Any | None
    rank_one_factored: Any | None
    chunked_recipe: Any | None

    def bind(
        self, candidate: LoweredCandidate, values: tuple[tuple[str, int], ...]
    ) -> TileLangExecutable:
        from ..lowering.chunked import select_chunked_schedule

        if candidate not in self.candidates:
            raise ValueError("candidate does not belong to this compilation")
        selected_plan = next(
            item
            for item in self.mathematical_plans
            if item.identity == candidate.plan_identity
        )
        values = tuple(values)
        kernel_config = KernelConfig(
            _digest((self.device_profile.target_identity, candidate.identity, values)),
            values,
        )
        compilation = self.compilation
        backward_binding = None
        backward_values = ()
        has_backward = self.backward_ir is not None and (
            candidate.lowerer == "tilelang-factored-recurrence"
            or not self.program._analysis.reference_required
        )
        if has_backward:
            from ..lowering.backward import bind_backward

            family = backward_context_family(
                self.program, candidate, self.dense_summary
            )
            backward_values = _backward_configuration(
                values, family, self.device_profile
            )
            backward_binding = bind_backward(self.backward_ir, family, backward_values)
        context_values = (
            tuple(ContextValue(*value) for value in backward_binding.context_values)
            if backward_binding is not None
            else ()
        )
        context_contract = ContextContract(context_values, context_values, ())
        context_contract.validate()
        toolchain = _toolchain_fingerprint()
        environment = _target_fingerprint(compilation.target)
        fixed_roles = tuple((str(role), size) for role, size in compilation.role_sizes)
        binary_identity = _digest(
            (
                candidate.identity,
                kernel_config.identity,
                context_contract.identity,
                compilation.mode,
                self.feature_sizes,
                fixed_roles,
                toolchain,
                environment,
            )
        )
        tuning_identity = _digest(
            (
                self.source_identity,
                fixed_roles,
                self.device_profile.target_identity,
                environment,
                toolchain,
                self.policy_version,
                compilation.mode,
                tuple(
                    (item.identity, kernel_configurations(item, self.device_profile))
                    for item in self.candidates
                ),
            )
        )
        workspace = WorkspacePolicy.for_candidate(
            candidate,
            self.estimates[candidate],
            compilation=compilation,
            algorithm=self.program,
            rank_one_factored=self.rank_one_factored,
        )
        backward_candidate = None
        backward_kernel_config = None
        if has_backward:
            backward_candidate = LoweredCandidate(
                _digest(
                    (self.source_identity, "tilelang-generic-backward", "backward")
                ),
                self.source_identity,
                "tilelang-generic-backward",
                "backward",
            )
            backward_kernel_config = KernelConfig(
                _digest(
                    (
                        backward_candidate.identity,
                        backward_values,
                        compilation.options.autotune,
                    )
                ),
                backward_values,
            )
        plan = TileLangPlan(
            self.source_identity,
            selected_plan,
            candidate,
            kernel_config,
            binary_identity,
            tuning_identity,
            compilation.mode,
            self.feature_sizes,
            context_contract,
            workspace,
            self.mathematical_plans,
            self.candidates,
            backward_candidate,
            backward_kernel_config,
            candidate_estimates=tuple(
                sorted(
                    ((item.identity, self.estimates[item]) for item in self.candidates),
                    key=lambda pair: (pair[1].cost, pair[0]),
                )
            ),
            estimation_shape=self.shape,
            device_profile=self.device_profile,
        )
        composition = Composition(candidate, backward_candidate, context_contract)
        composition.validate()
        rank_one = (
            self.rank_one_factored
            if candidate.lowerer == "tilelang-factored-recurrence"
            else None
        )
        return TileLangExecutable(
            plan=plan,
            backward_ir=self.backward_ir,
            dense_summary=self.dense_summary,
            scalar_factorized=self.scalar_factorized,
            program=self.program,
            rank_one_factored=rank_one,
            symmetric_rank_one=bool(
                rank_one is not None and rank_one.symmetric_rank_one
            ),
            composition=composition,
            backward_binding=backward_binding,
            chunked_recipe=select_chunked_schedule(
                self.chunked_recipe, candidate.lowerer
            ),
        )
