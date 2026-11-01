"""Typed composition of scans; child AlgorithmIRs retain their own scan proofs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence as TypingSequence
import math

import torch

from ..stateful_operator import (
    AlgorithmIR,
    Batch,
    CompiledStatefulOperator,
    ExecutionResult,
    FeatureRole,
    HeadRole,
    NamedOutputTuple,
    Sequence,
    Shift,
    StateContraction,
    StateTuple,
    StatefulCompilationError,
    StatefulOperator,
    Target,
    TensorInput,
    _ValidationRecipe,
    _canonical_identity,
    _expressions,
)

from .runtime.validation import validate_call
from .runtime.workspace import StageWorkspace


@dataclass(frozen=True)
class StageBinding:
    """Bind a downstream Input to an upstream Readout, with no dtype conversion."""

    input: str
    stage: str
    output: str


@dataclass(frozen=True)
class ScanStage:
    name: str
    algorithm: AlgorithmIR
    bindings: tuple[StageBinding, ...] = ()

    def __init__(self, name, algorithm, bindings=()):
        if not isinstance(name, str) or not name.isidentifier():
            raise StatefulCompilationError("IR_SCHEMA", "stages.name", {})
        from .logaddexp import LogAddExpScanIR

        if not isinstance(algorithm, (AlgorithmIR, LogAddExpScanIR)):
            raise StatefulCompilationError("IR_SCHEMA", f"stages.{name}.algorithm", {})
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "bindings", tuple(bindings))


@dataclass(frozen=True)
class StageOutput:
    name: str
    stage: str
    output: str


def _output_spec(stage, name):
    from .logaddexp import LogAddExpScanIR

    if isinstance(stage.algorithm, LogAddExpScanIR):
        return stage.algorithm.output_spec(name)
    algorithm = stage.algorithm._analysis.normalized
    for index, readout in enumerate(algorithm.readouts):
        if readout.name != name:
            continue
        if isinstance(readout, StateContraction):
            state = next(s for s in algorithm.states if s.name == readout.state)
            features = tuple(r for r in state.feature_roles if r != readout.role)
        else:
            types = dict(algorithm._analysis.expression_types)
            typed = types[f"readouts[{index}].expression"]
            features = tuple(r for r in typed.roles if isinstance(r, FeatureRole))
        if len(features) > 1:
            raise StatefulCompilationError(
                "IR_TYPE",
                f"stages.{stage.name}.outputs.{name}",
                {"expected": "scalar or vector readout"},
            )
        return TensorInput(
            name,
            (Batch, algorithm.states[0].head_role, Sequence, *features),
            readout.output_dtype,
        )
    raise StatefulCompilationError(
        "IR_TYPE", f"stages.{stage.name}.outputs", {"unknown": name}
    )


@dataclass(frozen=True, init=False)
class ScanStageIR:
    """A DAG of separately scanable AlgorithmIRs and typed trajectory bindings.

    State names are globally unique. Stage declaration order controls public
    State order; execution order is topological. Only exported outputs appear
    in ExecutionResult. A binding uses the author's downstream TensorInput
    roles and requires matching logical roles and dtype, including Head role.
    Different State Head roles across stages use existing child HeadMapping.
    """

    stages: tuple[ScanStage, ...]
    outputs: tuple[StageOutput, ...]
    inputs: tuple[TensorInput, ...]
    order: tuple[str, ...]

    def __init__(
        self, stages: TypingSequence[ScanStage], outputs: TypingSequence[StageOutput]
    ):
        stages, outputs = tuple(stages), tuple(outputs)
        if not stages or not outputs:
            raise StatefulCompilationError(
                "IR_SCHEMA", "stages", {"expected": "stages and outputs"}
            )
        by_name = {s.name: s for s in stages}
        if len(by_name) != len(stages):
            raise StatefulCompilationError(
                "IR_SCHEMA", "stages", {"reason": "duplicate name"}
            )
        states = [s.name for stage in stages for s in stage.algorithm.states]
        if len(set(states)) != len(states):
            raise StatefulCompilationError(
                "IR_SCHEMA", "states", {"reason": "duplicate name across stages"}
            )
        deps, external = {}, {}
        for stage in stages:
            declared = {s.name: s for s in stage.algorithm.inputs}
            bound = set()
            deps[stage.name] = set()
            for binding in stage.bindings:
                path = f"stages.{stage.name}.bindings.{binding.input}"
                if binding.input not in declared or binding.input in bound:
                    raise StatefulCompilationError(
                        "IR_TYPE", path, {"reason": "unknown or duplicate input"}
                    )
                source = by_name.get(binding.stage)
                if source is None:
                    raise StatefulCompilationError(
                        "IR_DEPENDENCY", path, {"unknown_stage": binding.stage}
                    )
                spec, wanted = (
                    _output_spec(source, binding.output),
                    declared[binding.input],
                )
                if spec.dtype != wanted.dtype or set(spec.roles) != set(wanted.roles):
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        path,
                        {
                            "expected_roles": list(map(str, spec.roles)),
                            "expected_dtype": str(spec.dtype),
                        },
                    )
                bound.add(binding.input)
                deps[stage.name].add(binding.stage)
            for name, spec in declared.items():
                if name in bound:
                    continue
                if name in external and external[name] != spec:
                    raise StatefulCompilationError(
                        "IR_TYPE",
                        f"inputs.{name}",
                        {"reason": "inconsistent declarations"},
                    )
                external[name] = spec
        order = []
        pending = {name: set(values) for name, values in deps.items()}
        while pending:
            ready = [name for name, values in pending.items() if not values]
            if not ready:
                raise StatefulCompilationError(
                    "IR_DEPENDENCY", "stages", {"cycle": sorted(pending)}
                )
            for name in ready:
                order.append(name)
                del pending[name]
            for values in pending.values():
                values.difference_update(ready)
        names = set()
        for output in outputs:
            if output.name in names or output.stage not in by_name:
                raise StatefulCompilationError(
                    "IR_TYPE", "outputs", {"name": output.name}
                )
            names.add(output.name)
            _output_spec(by_name[output.stage], output.output)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "inputs", tuple(external.values()))
        object.__setattr__(self, "order", tuple(order))

    @property
    def states(self):
        return tuple(s for stage in self.stages for s in stage.algorithm.states)

    @property
    def _structural_identity(self):
        return _canonical_identity(self)


class StagedStatefulOperator:
    def __init__(self, algorithm: ScanStageIR):
        self.algorithm = algorithm

    def __call__(self, **kwargs):
        raise RuntimeError(
            "StatefulOperator must be explicitly compiled before invocation"
        )

    def compile(
        self,
        *,
        role_sizes,
        target: Target,
        mode,
        optimized=True,
        autotune=False,
        backward=False,
        checkpoint=False,
    ):
        if not isinstance(checkpoint, bool):
            raise StatefulCompilationError("COMPILE_OPTIONS", "checkpoint", {})
        declared = {
            role
            for stage in self.algorithm.stages
            for spec in (*stage.algorithm.inputs, *stage.algorithm.states)
            for role in spec.roles
            if isinstance(role, (HeadRole, FeatureRole))
        }
        if set(role_sizes) != declared:
            raise StatefulCompilationError(
                "COMPILE_ROLE_SIZE",
                "role_sizes",
                {
                    "missing": list(map(str, declared - set(role_sizes))),
                    "unknown": list(map(str, set(role_sizes) - declared)),
                },
            )
        children = {}
        for stage in self.algorithm.stages:
            roles = {
                role
                for spec in (*stage.algorithm.inputs, *stage.algorithm.states)
                for role in spec.roles
                if isinstance(role, (HeadRole, FeatureRole))
            }
            from .logaddexp import LogAddExpOperator, LogAddExpScanIR

            operator_type = (
                LogAddExpOperator
                if isinstance(stage.algorithm, LogAddExpScanIR)
                else StatefulOperator
            )
            children[stage.name] = operator_type(stage.algorithm).compile(
                role_sizes={r: role_sizes[r] for r in roles},
                target=target,
                mode=mode,
                optimized=optimized,
                autotune=autotune,
                backward=backward,
            )
        return CompiledScanStages(
            self, children, role_sizes, target, mode, backward, checkpoint
        )


class CompiledScanStages:
    """Window dense inference trajectories; Torch autograd composes stage VJPs."""

    def __init__(
        self,
        operator,
        children: Mapping[str, CompiledStatefulOperator],
        role_sizes,
        target,
        mode,
        backward,
        checkpoint,
    ):
        self.operator = operator
        self.children = dict(children)
        self.role_sizes = dict(role_sizes)
        self.target, self.mode, self.backward = target, mode, backward
        self.checkpoint = checkpoint
        # Shift reads sequence-local input history, which mathematical States do
        # not carry across invocation boundaries.
        windowable = not any(
            isinstance(expression, Shift)
            for stage in operator.algorithm.stages
            for expression in _expressions(stage.algorithm)
        )
        self.specialization = _canonical_identity(
            (
                operator.algorithm,
                role_sizes,
                target,
                mode,
                backward,
                checkpoint,
                tuple(c.specialization for c in children.values()),
            )
        )
        self.runtime_family = "scan-stage-dag"
        recipes = [child.compilation.validation for child in self.children.values()]
        inputs = {}
        for stage in operator.algorithm.stages:
            bound_names = {binding.input for binding in stage.bindings}
            for spec in self.children[stage.name].compilation.validation.inputs:
                if spec.name not in bound_names:
                    inputs[spec.name] = spec
        self.validation = _ValidationRecipe(
            tuple(inputs),
            tuple(s.name for s in operator.algorithm.states),
            tuple(o.name for o in operator.algorithm.outputs),
            tuple(inputs.values()),
            tuple(s for recipe in recipes for s in recipe.states),
            True,
            False,
            (),
            allow_empty_dense=all(recipe.allow_empty_dense for recipe in recipes),
        )
        # Count expression tensors as well as trajectory copies. Each typed DAG
        # node is charged even if execution can reuse it: a conservative baseline.
        materialized_bytes_per_token = 0
        for stage in operator.algorithm.stages:
            head = role_sizes[stage.algorithm.states[0].head_role]
            expression_types = (
                stage.algorithm._analysis.expression_types
                if isinstance(stage.algorithm, AlgorithmIR)
                else ()
            )
            for _, typed in expression_types:
                elements = head * math.prod(
                    role_sizes[r] for r in typed.roles if isinstance(r, FeatureRole)
                )
                materialized_bytes_per_token += elements * 4
            for readout in stage.algorithm.readouts:
                spec = _output_spec(stage, readout.name)
                materialized_bytes_per_token += (
                    head
                    * math.prod(
                        role_sizes[r] for r in spec.roles if isinstance(r, FeatureRole)
                    )
                    * 8
                )

        stages = {stage.name: stage for stage in operator.algorithm.stages}
        output_bytes_per_token = sum(
            4
            * math.prod(
                role_sizes[role]
                for role in _output_spec(stages[output.stage], output.output).roles
                if isinstance(role, (HeadRole, FeatureRole))
            )
            for output in operator.algorithm.outputs
        )
        state_elements = sum(
            role_sizes[state.head_role]
            * math.prod(role_sizes[r] for r in state.feature_roles)
            for state in operator.algorithm.states
        )
        self._workspace = StageWorkspace(
            maximum_bytes=target.workspace_bytes,
            materialized_bytes_per_token=materialized_bytes_per_token,
            state_bytes_per_sequence=state_elements * 12,
            output_bytes_per_token=output_bytes_per_token,
            children=tuple(
                (child.workspace_bytes, isinstance(child, CompiledStatefulOperator))
                for child in self.children.values()
            ),
            mode=mode,
            backward=backward,
            windowable=windowable,
        )

    def workspace_bytes(self, token_count, *, sequence_count=1, sequence_length=None):
        return self._workspace.schedule(
            token_count,
            sequence_count=sequence_count,
            sequence_length=sequence_length,
        ).required_bytes

    def __call__(
        self,
        *,
        initial_state=None,
        sequence_offsets=None,
        return_final_state=False,
        **inputs,
    ):
        runtime, sizes, states, _ = validate_call(
            inputs,
            initial_state=initial_state,
            sequence_offsets=sequence_offsets,
            role_sizes=tuple(self.role_sizes.items()),
            validation=self.validation,
            mode=self.mode,
            target=self.target,
            backward=self.backward,
        )
        schedule = self._workspace.schedule(
            runtime.token_count,
            sequence_count=runtime.sequence_count,
            sequence_length=runtime.uniform_length,
        )
        if schedule.window_tokens and runtime.uniform_length > schedule.window_tokens:
            return self._execute_windows(
                inputs, states, runtime, schedule, return_final_state
            )
        return self._execute_stages(
            inputs, states, runtime, sequence_offsets, return_final_state
        )

    def _execute_windows(self, inputs, states, runtime, schedule, return_final_state):
        """Finish the DAG per budgeted window and carry every stage's State."""

        def execute_window(part, initial, segment):
            result = self._execute_stages(part, initial, segment, None, True)
            return result.outputs.values, result.final_state.as_dict()

        outputs, states = schedule.execute(
            inputs,
            {spec.name: spec.sequence_axis for spec in self.validation.inputs},
            runtime,
            states,
            execute_window,
            contiguous_inputs=True,
        )
        return ExecutionResult(
            NamedOutputTuple(self.validation.output_names, outputs),
            StateTuple(
                self.validation.state_names,
                tuple(states[name] for name in self.validation.state_names),
            )
            if return_final_state
            else None,
        )

    def _execute_stages(
        self, inputs, states, runtime, sequence_offsets, return_final_state
    ):
        stages = {s.name: s for s in self.operator.algorithm.stages}
        exports = {(o.stage, o.output) for o in self.operator.algorithm.outputs}
        remaining = {}
        for stage in stages.values():
            for binding in stage.bindings:
                key = (binding.stage, binding.output)
                remaining[key] = remaining.get(key, 0) + 1
        results, finals = {}, {}
        for name in self.operator.algorithm.order:
            stage, child = stages[name], self.children[name]
            bindings = {b.input: b for b in stage.bindings}
            bound = {}
            for spec in stage.algorithm.inputs:
                if spec.name not in bindings:
                    bound[spec.name] = inputs[spec.name]
                    continue
                binding = bindings[spec.name]
                key = (binding.stage, binding.output)
                value = results[key]
                source = _output_spec(stages[binding.stage], binding.output)
                if not any(isinstance(r, FeatureRole) for r in source.roles):
                    value = value.squeeze(-1)
                if runtime.packed:
                    source_roles = tuple(
                        r for r in source.roles if r not in (Batch, Sequence)
                    )
                    wanted_roles = tuple(
                        r for r in spec.roles if r not in (Batch, Sequence)
                    )
                    value = value.permute(
                        0, *(1 + source_roles.index(r) for r in wanted_roles)
                    )
                else:
                    value = value.permute(*(source.roles.index(r) for r in spec.roles))
                bound[spec.name] = value.contiguous()
                remaining[key] -= 1
            state_names = tuple(s.name for s in stage.algorithm.states)
            result = _invoke_stage(
                child,
                bound,
                StateTuple(state_names, tuple(states[s] for s in state_names)),
                sequence_offsets,
                return_final_state,
                checkpoint=self.checkpoint
                and self.backward
                and torch.is_grad_enabled(),
            )
            for output_name, tensor in result.outputs.as_dict().items():
                key = (name, output_name)
                if remaining.get(key, 0) or key in exports:
                    results[key] = tensor
            if result.final_state is not None:
                finals.update(result.final_state.as_dict())
            for key in tuple(results):
                if not remaining.get(key, 0) and key not in exports:
                    del results[key]
        return ExecutionResult(
            NamedOutputTuple(
                tuple(o.name for o in self.operator.algorithm.outputs),
                tuple(
                    results[(o.stage, o.output)]
                    for o in self.operator.algorithm.outputs
                ),
            ),
            StateTuple(
                self.validation.state_names,
                tuple(finals[s] for s in self.validation.state_names),
            )
            if return_final_state
            else None,
        )


def _invoke_stage(child, inputs, initial, offsets, return_final, *, checkpoint):
    if not checkpoint:
        return child(
            **inputs,
            initial_state=initial,
            sequence_offsets=offsets,
            return_final_state=return_final,
        )
    from torch.utils.checkpoint import checkpoint as recompute

    names = tuple(inputs)
    output_names = child.compilation.validation.output_names
    state_names = initial.names

    def execute(*tensors):
        result = child(
            **dict(zip(names, tensors[: len(names)], strict=True)),
            initial_state=StateTuple(state_names, tensors[len(names) :]),
            sequence_offsets=offsets,
            return_final_state=return_final,
        )
        return (*result.outputs, *(result.final_state or ()))

    tensors = recompute(
        execute,
        *inputs.values(),
        *initial,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    count = len(output_names)
    return ExecutionResult(
        NamedOutputTuple(output_names, tensors[:count]),
        StateTuple(state_names, tensors[count:]) if return_final else None,
    )
