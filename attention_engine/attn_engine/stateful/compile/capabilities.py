"""Specialization facts and lowerer-owned admission requirements.

Validation and candidate proposals share the same lazy structural recipes. Facts
contain no runtime tensors; laziness preserves validation's existing error order.
The registry attaches implementation requirements to lowerers, not the planner.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any


@dataclass(frozen=True)
class LoweringFacts:
    program: Any
    role_sizes: tuple[tuple[Any, int], ...]
    target: Any
    mode: str
    options: Any

    @cached_property
    def chunked_recipe(self):
        if self.mode != "dense":
            return None
        from ..lowering.chunked import compile_chunked_recipe

        return compile_chunked_recipe(self.program, dict(self.role_sizes))

    @property
    def backward_ir(self):
        return self.program._analysis.backward if self.options.backward else None

    @property
    def dense_inference(self):
        return (
            self.mode == "dense"
            and self.backward_ir is None
            and self.target.device_type == "cuda"
        )

    @property
    def native_diagonal(self):
        recipe = self.chunked_recipe
        return recipe.native_diagonal if recipe is not None else None

    @property
    def native_diagonal_available(self):
        return (
            self.options.optimized
            and not self.options.backward
            and self.native_diagonal is not None
        )

    @cached_property
    def _eligibility(self):
        return {}

    def eligibility(self, family):
        """Evaluate each requested structural proof at most once, not eagerly."""
        if family not in self._eligibility:
            from ..lowering.structured import RUNTIME_FAMILY_ELIGIBILITY

            rule = RUNTIME_FAMILY_ELIGIBILITY.get(family)
            self._eligibility[family] = (
                rule(self.program) if rule is not None else (True, None)
            )
        return self._eligibility[family]


def vector_propagation_supported(facts: LoweringFacts) -> bool:
    return facts.dense_inference


def scalar_factorized_supported(facts: LoweringFacts) -> bool:
    profile = facts.program._analysis.scalar_factorized
    # Only the fused kernel is priced; summary fallback is not this candidate.
    return (
        facts.backward_ir is None
        and profile is not None
        and profile.dense is not None
        and profile.injection_count == 1
        and profile.readout_count == 1
    )


def factored_backward_supported(facts: LoweringFacts) -> bool:
    if facts.backward_ir is None:
        return True
    profile = facts.program._analysis.rank_one_factored
    return profile is not None and all(
        operation.kind in {"column-scale", "row-scale", "scale", "rank-one"}
        for operation in profile.operations
    )


def shared_scores_supported(facts: LoweringFacts) -> bool:
    native = facts.native_diagonal
    return (
        native is not None
        and native.softmax is None
        and all(len(state.injections) == 1 for state in native.states)
        and any(state.identity_columns for state in native.states)
    )


def readout_cut_supported(facts: LoweringFacts) -> bool:
    native = facts.native_diagonal
    return native is not None and native.softmax is not None and native.fused_softmax
