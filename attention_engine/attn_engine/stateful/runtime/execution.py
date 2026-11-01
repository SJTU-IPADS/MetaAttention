"""Execution bound to one compiler-selected lowering, including legal fallbacks.

The public operator owns validation. This module owns the selected adapter,
input preparation, bounded replay and result assembly; it never selects a new
Mathematical Plan from runtime dimensions.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ...stateful_operator import (
    AxisScale,
    Batch,
    ExecutionResult,
    Identity,
    LogicalRole,
    NamedOutputTuple,
    One,
    RankOnePropagation,
    Sequence,
    StateContraction,
    StateTuple,
    StatefulCompilationError,
    _Runtime,
    _ValidationRecipe,
    _evaluate,
    _feature_vector,
    _head_role,
    _segmented_affine_scan,
    _segmented_elementwise_scan,
    _token_major,
)


class SelectedExecution:
    """A selected lowering's execution contract, not a second plan selector."""

    def __init__(self, program, executable):
        self._program = program
        self.executable = executable
        family = executable.runtime_family
        self._reference = family == "stateful-reference"
        self._reference_required = program._analysis.reference_required
        self._factored = executable.rank_one_factored is not None
        self._scalar = family == "scalar-factorized"
        self._product_injection = family == "product-injection-gemm"
        self._shared_readout = family == "shared-state-contraction"
        self._summary_family = (
            "axis-scale" if family == "vector-propagation" else family
        )
        self._scan = (
            self._lower_factored_recurrence
            if self._factored
            else self._lower_vector_propagation
            if family == "vector-propagation"
            else self._lower_parallel_scan
        )
        profile = executable.scalar_factorized
        self._raw_scalar = (
            self._scalar and profile is not None and profile.dense is not None
        )
        self._dense_inference = None
        recipe = executable.chunked_recipe
        if recipe is not None and family in {
            "chunked-stateful",
            "product-feature-chunked",
            "rank-one-diagonal",
            "native-normalized",
            "native-diagonal",
        }:
            if family == "rank-one-diagonal":
                if recipe.rank_one is not None:
                    self._dense_inference = self._execute_rank_one
            elif (
                family != "native-normalized" or recipe.native_normalized is not None
            ) and (family != "native-diagonal" or recipe.native_diagonal is not None):
                from ..lowering.chunked import bind_chunked_executor

                self._dense_inference = self._execute_chunked_stateful
                self._chunked_execute = bind_chunked_executor(recipe)

    def __call__(
        self,
        *,
        runtime,
        role_sizes,
        inputs,
        initial_states,
        return_final_state,
        validation,
        schedule,
        requires_grad,
    ):
        executable = self.executable
        cuda = runtime.device.type == "cuda"
        if (
            self._dense_inference is not None
            and cuda
            and not runtime.packed
            and runtime.token_count > 0
            and not requires_grad
        ):
            return self._dense_inference(
                runtime,
                role_sizes,
                inputs,
                initial_states,
                return_final_state,
                executable.chunked_recipe,
                validation,
            )
        if self._reference or (
            self._reference_required
            and not (self._factored and cuda and runtime.token_count > 0)
        ):
            return self._execute_reference(
                runtime, role_sizes, inputs, initial_states, return_final_state
            )
        state = initial_states[self._program.states[0].name]
        if self._raw_scalar and cuda and not runtime.packed and not requires_grad:
            assert runtime.uniform_length is not None
            output, final = executable.scalar_factorized_dense_raw(
                inputs,
                state,
                sequence_count=runtime.sequence_count,
                sequence_length=runtime.uniform_length,
                output_dtype=self._program.readouts[0].output_dtype,
                return_final_state=return_final_state,
            )
            outputs = (output,)
            final_state = (
                StateTuple((self._program.states[0].name,), (final,))
                if return_final_state
                else None
            )
        elif (
            schedule.window_tokens
            and cuda
            and not runtime.packed
            and runtime.uniform_length > schedule.window_tokens
        ):
            outputs, final_state = self._execute_factored_windows(
                inputs,
                state,
                runtime,
                role_sizes,
                return_final_state,
                executable,
                schedule,
            )
        else:
            values = self._normalize_inputs(inputs, runtime, role_sizes)
            # Empty packed scans and CPU reference summaries do not enter CUDA adapters.
            scan = (
                self._scan
                if cuda and runtime.token_count > 0
                else self._lower_parallel_scan
            )
            outputs, final_state = scan(
                values, state, runtime, return_final_state, executable
            )
        return ExecutionResult(
            NamedOutputTuple(validation.output_names, outputs), final_state
        )

    def _execute_rank_one(
        self,
        runtime,
        role_sizes,
        inputs,
        initial_states,
        return_final_state,
        recipe,
        validation,
    ):
        from ..lowering.scan.rank1 import execute_rank_one

        output, final = execute_rank_one(
            recipe.rank_one,
            inputs,
            initial_states[recipe.rank_one.state],
            sequence_count=runtime.sequence_count,
            sequence_length=runtime.uniform_length,
            heads=runtime.state_heads,
            return_final_state=return_final_state,
        )
        return ExecutionResult(
            NamedOutputTuple._unchecked(validation.output_names, (output,)),
            StateTuple._unchecked(validation.state_names, (final,))
            if final is not None
            else None,
        )

    def _execute_factored_windows(
        self, inputs, state, runtime, role_sizes, return_final_state, plan, schedule
    ):
        """Prepare/replay one budgeted window and carry the mathematical State."""
        state_name = self._program.states[0].name

        def execute_window(part, initial, segment):
            values = self._normalize_inputs(part, segment, role_sizes)
            outputs, final = self._lower_factored_recurrence(
                values, initial, segment, True, plan
            )
            return outputs, final[state_name]

        outputs, state = schedule.execute(
            inputs,
            {
                spec.name: spec.roles.index(Sequence)
                if Sequence in spec.roles
                else None
                for spec in self._program.inputs
            },
            runtime,
            state,
            execute_window,
        )
        final_state = (
            StateTuple((state_name,), (state,)) if return_final_state else None
        )
        return outputs, final_state

    def _execute_chunked_stateful(
        self,
        runtime: _Runtime,
        role_sizes: Mapping[LogicalRole, int],
        inputs: Mapping[str, torch.Tensor],
        initial_states: Mapping[str, torch.Tensor],
        return_final_state: bool,
        recipe: Any,
        validation: _ValidationRecipe,
    ) -> ExecutionResult:
        """Execute compiler-proven dense inference without reference plumbing."""
        from ..lowering.chunked import bind_chunked_inputs

        assert runtime.uniform_length is not None
        if recipe.rank_one is not None:
            raise RuntimeError("rank-one recipe is not selected by the plan")
        values = bind_chunked_inputs(recipe, inputs, runtime)
        result = self._chunked_execute(
            self._program,
            values=values,
            initial_states=initial_states,
            segment=None,
            role_sizes=role_sizes,
            heads=runtime.state_heads,
            token_count=runtime.token_count,
            device=runtime.device,
            return_final_state=return_final_state,
            sequence_length=runtime.uniform_length,
        )
        assert result is not None
        outputs, finals = result
        ordered = tuple(
            outputs[readout.name]
            .reshape(
                runtime.sequence_count, runtime.uniform_length, runtime.state_heads, -1
            )
            .permute(0, 2, 1, 3)
            for readout in self._program.readouts
        )
        return ExecutionResult(
            NamedOutputTuple._unchecked(validation.output_names, ordered),
            StateTuple._unchecked(
                validation.state_names,
                tuple(finals[name] for name in validation.state_names),
            )
            if finals is not None
            else None,
        )

    def _execute_reference(
        self,
        runtime: _Runtime,
        role_sizes: Mapping[LogicalRole, int],
        inputs: Mapping[str, torch.Tensor],
        initial_states: Mapping[str, torch.Tensor],
        return_final_state: bool,
    ) -> ExecutionResult:
        """Execute a schema-v2 Program through the reference lowering."""
        from ..lowering.reference import execute_reference, normalize_inputs

        algorithm = self._program
        heads = runtime.state_heads
        if runtime.packed:
            assert runtime.offsets is not None
            lengths = runtime.offsets[1:] - runtime.offsets[:-1]
            segment = torch.repeat_interleave(
                torch.arange(runtime.sequence_count, device=runtime.device),
                lengths.long(),
            )
        else:
            assert runtime.uniform_length is not None
            segment = torch.arange(
                runtime.sequence_count, device=runtime.device
            ).repeat_interleave(runtime.uniform_length)
        values = normalize_inputs(
            algorithm,
            role_sizes,
            inputs,
            packed=runtime.packed,
            sequence_count=runtime.sequence_count,
            token_count=runtime.token_count,
            uniform_length=runtime.uniform_length,
            state_heads=heads,
            device=runtime.device,
        )
        outputs, finals = execute_reference(
            algorithm,
            values=values,
            initial_states=initial_states,
            segment=segment,
            role_sizes=role_sizes,
            heads=heads,
            token_count=runtime.token_count,
            device=runtime.device,
            return_final_state=return_final_state,
            sequence_length=(runtime.uniform_length if not runtime.packed else None),
        )
        ordered: list[torch.Tensor] = []
        for readout in algorithm.readouts:
            tensor = outputs[readout.name]
            if runtime.packed:
                ordered.append(tensor)
            else:
                assert runtime.uniform_length is not None
                ordered.append(
                    tensor.reshape(
                        runtime.sequence_count,
                        runtime.uniform_length,
                        heads,
                        tensor.shape[-1],
                    ).permute(0, 2, 1, 3)
                )
        final_state = None
        if finals is not None:
            final_state = StateTuple(
                tuple(spec.name for spec in algorithm.states),
                tuple(finals[spec.name] for spec in algorithm.states),
            )
        return ExecutionResult(
            NamedOutputTuple(
                tuple(readout.name for readout in algorithm.readouts), tuple(ordered)
            ),
            final_state,
        )

    def _normalize_inputs(
        self,
        inputs: Mapping[str, torch.Tensor],
        runtime: _Runtime,
        role_sizes: Mapping[str, int],
    ) -> dict[str, torch.Tensor]:
        state = self._program.states[0]
        features = state.feature_roles
        result: dict[str, torch.Tensor] = {}
        for spec in self._program.inputs:
            tensor = inputs[spec.name]
            if runtime.packed and Sequence in spec.roles:
                physical_roles = tuple(
                    role for role in spec.roles if role not in (Batch, Sequence)
                )
                value = tensor
                token_axis = 0
            elif Sequence in spec.roles:
                head_role = _head_role(spec.roles, features)
                order = [spec.roles.index(Batch), spec.roles.index(Sequence)]
                if head_role is not None:
                    order.append(spec.roles.index(head_role))
                order.extend(
                    spec.roles.index(role) for role in features if role in spec.roles
                )
                value = tensor.permute(order).reshape(
                    runtime.token_count, *[tensor.shape[index] for index in order[2:]]
                )
                physical_roles = tuple(spec.roles[index] for index in order[2:])
                token_axis = 0
            else:
                batch_axis = spec.roles.index(Batch) if Batch in spec.roles else None
                singleton_axis = (
                    batch_axis if batch_axis is not None else spec.roles.index(One)
                )
                head_role = _head_role(spec.roles, features)
                order = [singleton_axis]
                if head_role is not None:
                    order.append(spec.roles.index(head_role))
                order.extend(
                    spec.roles.index(role) for role in features if role in spec.roles
                )
                value = tensor.permute(order)
                if batch_axis is None:
                    value = value.reshape(1, *value.shape[1:]).expand(
                        runtime.sequence_count, *value.shape[1:]
                    )
                value = value.repeat_interleave(runtime.uniform_length, dim=0)
                physical_roles = tuple(
                    spec.roles[index]
                    for index in order
                    if spec.roles[index] not in (Batch, One)
                )
                token_axis = 0
            head_role = _head_role(physical_roles, features)
            if token_axis is None:
                if head_role is None:
                    value = value.reshape(1, 1, *value.shape[len(spec.roles) :])
                else:
                    head_axis = physical_roles.index(head_role)
                    value = value.movedim(head_axis, 0).unsqueeze(0)
                value = value.expand(runtime.token_count, *value.shape[1:])
            elif head_role is None:
                value = value.unsqueeze(1)
            else:
                head_axis = 1 + physical_roles.index(head_role)
                if head_axis != 1:
                    value = value.movedim(head_axis, 1)
            source_heads = value.shape[1]
            if source_heads != runtime.state_heads:
                if runtime.state_heads % source_heads:
                    raise StatefulCompilationError(
                        "RUNTIME_SHAPE", f"inputs.{spec.name}", {"role": head_role}
                    )
                value = value.repeat_interleave(
                    runtime.state_heads // source_heads, dim=1
                )
            present_features = tuple(role for role in features if role in spec.roles)
            if present_features == ():
                value = value.reshape(runtime.token_count, runtime.state_heads, 1, 1)
            elif present_features == (features[0],):
                value = value.reshape(
                    runtime.token_count,
                    runtime.state_heads,
                    runtime.feature_sizes[0],
                    1,
                )
            elif present_features == (features[1],):
                value = value.reshape(
                    runtime.token_count,
                    runtime.state_heads,
                    1,
                    runtime.feature_sizes[1],
                )
            else:
                value = value.reshape(
                    runtime.token_count,
                    runtime.state_heads,
                    runtime.feature_sizes[0],
                    runtime.feature_sizes[1],
                )
            result[spec.name] = value
        return result

    def _lower_parallel_scan(
        self,
        values: Mapping[str, torch.Tensor],
        initial_state: torch.Tensor,
        runtime: _Runtime,
        return_final_state: bool,
        executable: Any,
    ) -> tuple[tuple[torch.Tensor, ...], StateTuple | None]:
        token_count = runtime.token_count
        heads = runtime.state_heads
        left_size, right_size = runtime.feature_sizes
        device = runtime.device
        state_spec = self._program.states[0]
        if runtime.packed:
            assert runtime.offsets is not None
            lengths = runtime.offsets[1:] - runtime.offsets[:-1]
            segment = torch.repeat_interleave(
                torch.arange(runtime.sequence_count, device=device), lengths.long()
            )
        else:
            assert runtime.uniform_length is not None
            segment = torch.arange(
                runtime.sequence_count, device=device
            ).repeat_interleave(runtime.uniform_length)
        if token_count == 0:
            outputs = tuple(
                torch.empty(
                    (
                        0,
                        heads,
                        right_size
                        if readout.role == state_spec.feature_roles[0]
                        else left_size,
                    ),
                    device=device,
                    dtype=readout.output_dtype,
                )
                for readout in self._program.readouts
            )
            final = (
                StateTuple((state_spec.name,), (initial_state,))
                if return_final_state
                else None
            )
            return outputs, final
        from ..lowering.structured import (
            _axis_scale_summary,
            product_injection_terms,
            rank_one_summary,
            shared_readouts,
        )

        family = self._summary_family
        identity_only = not self._program.transition.propagations
        needs_dense_transform = bool(getattr(executable, "dense_summary", False))
        second_factor: torch.Tensor | None = None
        if family == "axis-scale":
            left, right, elementwise, second_factor = _axis_scale_summary(
                self._program.transition.propagations,
                evaluate=_evaluate,
                values=values,
                state_spec=state_spec,
                token_count=token_count,
                heads=heads,
                left_size=left_size,
                right_size=right_size,
                device=device,
                dense=needs_dense_transform,
            )
            identity_left = left if needs_dense_transform else None
            identity_right = right if needs_dense_transform else None
        elif family == "rank-one-factorized":
            left, right, elementwise = rank_one_summary(
                self._program.transition.propagations,
                evaluate=_evaluate,
                feature_vector=_feature_vector,
                values=values,
                state_spec=state_spec,
                token_count=token_count,
                heads=heads,
                left_size=left_size,
                right_size=right_size,
                device=device,
            )
            identity_left = identity_right = None
        else:
            if needs_dense_transform or family in {
                "generic",
                "product-injection-gemm",
                "shared-state-contraction",
            }:
                identity_left = (
                    torch.eye(left_size, device=device)
                    .expand(token_count, heads, left_size, left_size)
                    .contiguous()
                )
                identity_right = (
                    torch.eye(right_size, device=device)
                    .expand(token_count, heads, right_size, right_size)
                    .contiguous()
                )
                left: torch.Tensor | None = identity_left
                right: torch.Tensor | None = identity_right
            else:
                identity_left = identity_right = None
                left = right = None
            elementwise: torch.Tensor | None = (
                torch.ones(token_count, heads, 1, 1, device=device)
                if identity_only
                else None
            )
            for node in self._program.transition.propagations:
                if isinstance(node, Identity):
                    continue
                if isinstance(node, AxisScale):
                    factor = _evaluate(node.factor, values).float()
                    if len(node.roles) == 2 or not node.roles:
                        if needs_dense_transform:
                            assert left is not None
                            left = factor * left
                        else:
                            elementwise = (
                                factor if elementwise is None else elementwise * factor
                            )
                    elif node.roles[0] == state_spec.feature_roles[0]:
                        assert left is not None
                        left = torch.diag_embed(factor[..., :, 0]) @ left
                    else:
                        assert right is not None
                        right = right @ torch.diag_embed(factor[..., 0, :])
                elif isinstance(node, RankOnePropagation):
                    left_vector = _feature_vector(
                        _evaluate(node.left, values), node.role, state_spec
                    )
                    right_vector = _feature_vector(
                        _evaluate(node.right, values), node.role, state_spec
                    )
                    coefficient = _evaluate(node.coefficient, values).float()
                    assert identity_left is not None and identity_right is not None
                    assert left is not None and right is not None
                    if node.role == state_spec.feature_roles[0]:
                        update = identity_left + coefficient * left_vector.unsqueeze(
                            -1
                        ) * right_vector.unsqueeze(-2)
                        left = update @ left
                    else:
                        update = identity_right + coefficient * right_vector.unsqueeze(
                            -1
                        ) * left_vector.unsqueeze(-2)
                        right = right @ update
        scalar_profile = getattr(executable, "scalar_factorized", None)

        can_fuse_scalar = (
            self._scalar
            and scalar_profile is not None
            and runtime.device.type == "cuda"
            and scalar_profile.injection_count == 1
            and scalar_profile.readout_count == 1
            and not any(tensor.requires_grad for tensor in values.values())
            and not initial_state.requires_grad
        )
        if can_fuse_scalar:
            injection_term = self._program.transition.injections[0]
            factors = {
                factor.roles[0]: _evaluate(factor.expression, values)
                for factor in injection_term.factors
            }
            dense_recipe = scalar_profile.dense
            assert dense_recipe is not None
            left_offset = dense_recipe.left_factor[1].offset
            right_offset = dense_recipe.right_factor[1].offset
            readout_offset = dense_recipe.readout[1].offset
            readout = _feature_vector(
                _evaluate(self._program.readouts[0].operand, values),
                self._program.readouts[0].role,
                state_spec,
            )
            if runtime.packed:
                assert runtime.offsets is not None
                max_sequence_length = max(runtime.token_count, 1)
                output, fused_final = executable.scalar_factorized_packed(
                    elementwise,
                    factors[state_spec.feature_roles[0]],
                    factors[state_spec.feature_roles[1]],
                    readout,
                    initial_state,
                    runtime.offsets,
                    sequence_count=runtime.sequence_count,
                    max_sequence_length=max_sequence_length,
                    output_dtype=self._program.readouts[0].output_dtype,
                    return_final_state=return_final_state,
                    left_offset=left_offset,
                    right_offset=right_offset,
                    readout_offset=readout_offset,
                )
            else:
                output, fused_final = executable.scalar_factorized_dense(
                    elementwise,
                    factors[state_spec.feature_roles[0]],
                    factors[state_spec.feature_roles[1]],
                    readout,
                    initial_state,
                    sequence_count=runtime.sequence_count,
                    sequence_length=runtime.uniform_length,
                    output_dtype=self._program.readouts[0].output_dtype,
                    return_final_state=return_final_state,
                    left_offset=left_offset,
                    right_offset=right_offset,
                    readout_offset=readout_offset,
                )
            output_values = [output]
            final_state = (
                StateTuple((state_spec.name,), (fused_final,))
                if return_final_state
                else None
            )
            return tuple(output_values), final_state

        if elementwise is not None and family == "axis-scale":
            # tried before the injection is assembled: the chunked schedule takes
            # the injection's own per-axis factors, so materializing the
            # tokens*heads*rows*columns product first (268 MB for a 64x64 State at
            # T=2048) is pure waste on this path
            chunked = self._lower_chunked_elementwise(
                values, initial_state, elementwise, second_factor, runtime, executable
            )
            if chunked is not None:
                output, chunked_final = chunked
                final_state = (
                    StateTuple((state_spec.name,), (chunked_final,))
                    if return_final_state
                    else None
                )
                return (output,), final_state
        if self._product_injection:
            injection = product_injection_terms(
                self._program.transition.injections,
                evaluate=_evaluate,
                values=values,
                feature_roles=state_spec.feature_roles,
                token_count=token_count,
                heads=heads,
                left_size=left_size,
                right_size=right_size,
                device=device,
            )
        else:
            injection = None
            for term in self._program.transition.injections:
                product: torch.Tensor | float = 1.0
                for factor in term.factors:
                    product = product * _evaluate(factor.expression, values).float()
                if injection is None:
                    injection = product.contiguous()
                else:
                    injection.add_(product)
        if injection is None:
            injection = torch.zeros(
                token_count, heads, left_size, right_size, device=device
            )
        segment_i32 = segment.to(torch.int32)
        if elementwise is not None:
            # the summary factors are derived from the token-major input views,
            # so they are not always contiguous; materializing them here is the
            # pre-existing seam this lowering declares (see the ticket note on
            # spec section "Implementation Decisions")
            scale = elementwise.contiguous()
            second_operand = (
                second_factor.contiguous() if second_factor is not None else None
            )
            if runtime.device.type == "cuda":
                states = executable.prefix(
                    scale,
                    injection,
                    segment_i32,
                    initial_state,
                    second=second_operand,
                )
            else:
                if second_operand is not None:
                    # the CPU reference composes the pair before the scan
                    scale = scale * second_operand
                scale, injection = _segmented_elementwise_scan(
                    scale, injection, segment
                )
                states = scale * initial_state[segment] + injection
        else:
            assert left is not None and right is not None
            if runtime.device.type == "cuda":
                states = executable.prefix(
                    (left, right), injection, segment_i32, initial_state
                )
            else:
                left, right, injection = _segmented_affine_scan(
                    left, right, injection, segment
                )
                states = left @ initial_state[segment] @ right + injection
        output_values: list[torch.Tensor] = []
        shared = (
            shared_readouts(
                self._program.readouts,
                evaluate=_evaluate,
                feature_vector=_feature_vector,
                values=values,
                states=states,
                state_spec=state_spec,
            )
            if self._shared_readout
            else {}
        )
        for readout in self._program.readouts:
            if readout.name in shared:
                output = shared[readout.name]
                if not runtime.packed:
                    assert runtime.uniform_length is not None
                    output = output.reshape(
                        runtime.sequence_count,
                        runtime.uniform_length,
                        heads,
                        output.shape[-1],
                    ).permute(0, 2, 1, 3)
                output_values.append(output)
                continue
            operand = _feature_vector(
                _evaluate(readout.operand, values), readout.role, state_spec
            ).float()
            if readout.role == state_spec.feature_roles[0]:
                output = torch.einsum("nhf,nhfg->nhg", operand, states)
            else:
                output = torch.einsum("nhg,nhfg->nhf", operand, states)
            output = output.to(readout.output_dtype)
            if not runtime.packed:
                assert runtime.uniform_length is not None
                output = output.reshape(
                    runtime.sequence_count,
                    runtime.uniform_length,
                    heads,
                    output.shape[-1],
                ).permute(0, 2, 1, 3)
            output_values.append(output)

        final_state = None
        if return_final_state:
            if runtime.packed:
                assert runtime.offsets is not None
                lengths = runtime.offsets[1:] - runtime.offsets[:-1]
                last = (runtime.offsets[1:] - 1).clamp_min(0).long()
                gathered = states[last]
                final = torch.where(
                    (lengths == 0).view(-1, 1, 1, 1), initial_state, gathered
                )
            else:
                assert runtime.uniform_length is not None
                final = states.reshape(
                    runtime.sequence_count,
                    runtime.uniform_length,
                    heads,
                    left_size,
                    right_size,
                )[:, -1]
            final_state = StateTuple((state_spec.name,), (final,))
        return tuple(output_values), final_state

    def _lower_vector_propagation(
        self,
        values: Mapping[str, torch.Tensor],
        initial_state: torch.Tensor,
        runtime: _Runtime,
        return_final_state: bool,
        executable: Any,
    ) -> tuple[tuple[torch.Tensor, ...], StateTuple | None]:
        if (
            runtime.device.type != "cuda"
            or runtime.packed
            or runtime.uniform_length is None
            or runtime.uniform_length * runtime.sequence_count != runtime.token_count
            or any(tensor.requires_grad for tensor in values.values())
            or initial_state.requires_grad
        ):
            raise RuntimeError("vector-propagation requires dense inference inputs")
        state_spec = self._program.states[0]
        first_role, second_role = state_spec.feature_roles
        transition = self._program.transition
        gate_expression = transition.propagations[0].factor
        if runtime.uniform_length >= 64:
            # Independent chunk replay also handles batches and nonaligned tails.
            chunked = self._lower_chunked_elementwise(
                values,
                initial_state,
                _evaluate(gate_expression.operand, values),
                None,
                runtime,
                executable,
                log_gate_rows=True,
            )
            if chunked is not None:
                output, final = chunked
                final_state = (
                    StateTuple((state_spec.name,), (final,))
                    if return_final_state
                    else None
                )
                return (output,), final_state
        injection = transition.injections[0]
        factors = {
            factor.roles[0]: _evaluate(factor.expression, values)
            for factor in injection.factors
        }
        gate = _feature_vector(
            _evaluate(gate_expression.operand, values), first_role, state_spec
        )
        readout = self._program.readouts[0]
        query = _feature_vector(
            _evaluate(readout.operand, values), readout.role, state_spec
        )
        key = _feature_vector(factors[first_role], first_role, state_spec)
        value = _feature_vector(factors[second_role], second_role, state_spec)
        output, final = executable.vector_propagation_dense(
            log_gate=_token_major(gate, runtime),
            key=_token_major(key, runtime),
            value=_token_major(value, runtime),
            query=_token_major(query, runtime),
            initial_state=initial_state,
            output_dtype=readout.output_dtype,
        )
        final_state = (
            StateTuple((state_spec.name,), (final,)) if return_final_state else None
        )
        return (output,), final_state

    def _lower_chunked_elementwise(
        self,
        values: Mapping[str, torch.Tensor],
        initial_state: torch.Tensor,
        elementwise: torch.Tensor,
        second_factor: torch.Tensor | None,
        runtime: _Runtime,
        executable: Any,
        *,
        log_gate_rows: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Run a diagonal recurrence through the chunked schedule.

        The kernels need one injection factor pair, one first-Feature readout,
        dense sequences, and no gradient. Tails use identity propagation and
        zero injections outside the logical sequence.
        """
        if (
            runtime.device.type != "cuda"
            or runtime.packed
            or runtime.uniform_length is None
            or runtime.uniform_length * runtime.sequence_count != runtime.token_count
            or len(self._program.transition.injections) != 1
            or len(self._program.readouts) != 1
            or any(tensor.requires_grad for tensor in values.values())
            or initial_state.requires_grad
        ):
            return None
        state_spec = self._program.states[0]
        first_role, second_role = state_spec.feature_roles
        injection_term = self._program.transition.injections[0]
        factors = {
            factor.roles[0]: _evaluate(factor.expression, values)
            for factor in injection_term.factors
        }
        if set(factors) != {first_role, second_role}:
            return None
        readout = self._program.readouts[0]
        if not isinstance(readout, StateContraction) or readout.role != first_role:
            return None
        outputs, finals = executable.chunked_elementwise_dense(
            gate_rows=(_token_major(elementwise, runtime),),
            gate_columns=(
                None if second_factor is None else _token_major(second_factor, runtime),
            ),
            key=(_token_major(factors[first_role], runtime),),
            value=(_token_major(factors[second_role], runtime),),
            query=(
                _token_major(
                    _feature_vector(
                        _evaluate(readout.operand, values), readout.role, state_spec
                    ),
                    runtime,
                ),
            ),
            initial_state=(initial_state,),
            output_dtype=readout.output_dtype,
            log_gate_rows=(log_gate_rows,),
            unit_gate_columns=(second_factor is None,),
        )
        # one State reaches this path today; the entry returns a per-State pair so
        # a readout that composes several States can consume them
        return outputs[0], finals[0]

    def _lower_factored_recurrence(
        self,
        values: Mapping[str, torch.Tensor],
        initial_state: torch.Tensor,
        runtime: _Runtime,
        return_final_state: bool,
        executable: Any,
    ) -> tuple[tuple[torch.Tensor, ...], StateTuple | None]:
        """Lower inference through chunks; retain the compiled backward path."""
        from ..lowering.lowrank.factored import factored_recurrence_chunked

        state_spec = self._program.states[0]
        first_role, second_role = state_spec.feature_roles
        recipe = executable.rank_one_factored
        assert recipe is not None
        if recipe.transpose_state:
            first_role, second_role = second_role, first_role
            initial_state = initial_state.transpose(-1, -2).contiguous()
        backward_path = (
            executable.backward_binding is not None
            and torch.is_grad_enabled()
            and (
                initial_state.requires_grad
                or any(value.requires_grad for value in values.values())
            )
        )
        if backward_path:
            token_count, heads = runtime.token_count, runtime.state_heads
            rows = initial_state.shape[-2]
            ones_row = torch.ones(token_count, heads, rows, 1, device=runtime.device)
            ones_scalar = torch.ones(token_count, heads, 1, 1, device=runtime.device)
        codes: list[str] = []
        left: list[torch.Tensor | None] = []
        right: list[torch.Tensor | None] = []
        coefficient: list[torch.Tensor | None] = []
        for node, operation in zip(
            self._program.transition.propagations, recipe.operations, strict=True
        ):
            if isinstance(node, AxisScale):
                if not node.roles:
                    codes.append(operation.kind)
                    left.append(None)
                    right.append(None)
                    coefficient.append(_evaluate(node.factor, values).float())
                elif tuple(node.roles) == (first_role,):
                    codes.append(operation.kind)
                    left.append(_evaluate(node.factor, values).float())
                    right.append(None)
                    coefficient.append(None)
                else:
                    codes.append(operation.kind)
                    left.append(_evaluate(node.factor, values).float())
                    right.append(None)
                    coefficient.append(None)
            else:
                codes.append(operation.kind)
                left.append(_evaluate(node.left, values).float())
                right.append(_evaluate(node.right, values).float())
                coefficient.append(_evaluate(node.coefficient, values).float())
        injection_left: list[torch.Tensor] = []
        injection_right: list[torch.Tensor] = []
        injection_scalars: list[torch.Tensor] = []
        injection_orders: list[tuple[int, ...]] = []
        for injection, write in zip(
            self._program.transition.injections, recipe.injections, strict=True
        ):
            factors = injection.factors
            injection_left.append(
                _evaluate(factors[write.left_factor].expression, values).float()
            )
            injection_right.append(
                _evaluate(factors[write.right_factor].expression, values).float()
            )
            if recipe.injection_propagations:
                # Convert an ordered write to its final additive contribution
                # by applying subsequent actions to its vector factors. This
                # is linear in suffix length, with no outer-product expansion.
                row = injection_left[-1].reshape(
                    runtime.token_count, runtime.state_heads, initial_state.shape[-2], 1
                )
                column = injection_right[-1].reshape(
                    runtime.token_count, runtime.state_heads, 1, initial_state.shape[-1]
                )
                for action_index in recipe.injection_propagations[
                    len(injection_left) - 1
                ]:
                    action = self._program.transition.propagations[action_index]
                    if isinstance(action, AxisScale):
                        factor = _evaluate(action.factor, values).float()
                        if not action.roles:
                            row = factor * row
                        elif action.roles == (first_role,):
                            row = factor.reshape_as(row) * row
                        else:
                            column = factor.reshape_as(column) * column
                    else:
                        action_left = (
                            _evaluate(action.left, values).float().reshape_as(row)
                        )
                        action_right = (
                            _evaluate(action.right, values).float().reshape_as(row)
                        )
                        scalar = _evaluate(action.coefficient, values).float()
                        row = row + scalar * action_left * (action_right * row).sum(
                            -2, keepdim=True
                        )
                injection_left[-1], injection_right[-1] = row, column
            scalar_base = len(injection_scalars)
            injection_scalars.extend(
                _evaluate(factors[index].expression, values).float()
                for index in write.scalar_factors
            )
            injection_orders.append(
                tuple(
                    index if index < 0 else scalar_base + index
                    for index in write.factor_order
                )
            )
        readouts: list[torch.Tensor] = []
        dtypes: list[torch.dtype] = []
        for readout in self._program.readouts:
            assert isinstance(readout, StateContraction)
            readouts.append(_evaluate(readout.operand, values).float())
            dtypes.append(readout.output_dtype)
        if runtime.packed:
            assert runtime.offsets is not None
            from ..lowering.packed import execute_lowrank_packed

            outputs, final = execute_lowrank_packed(
                operation_codes=tuple(codes),
                left_factors=left,
                right_factors=right,
                coefficients=coefficient,
                injection_left=injection_left,
                injection_right=injection_right,
                injection_scalars=injection_scalars,
                injection_orders=tuple(injection_orders),
                readout_axis=recipe.readout_axis,
                readouts=readouts,
                initial_state=initial_state,
                sequence_offsets=runtime.offsets,
                output_dtypes=tuple(dtypes),
                return_final_state=return_final_state,
            )
            if final is not None and recipe.transpose_state:
                final = final.transpose(-1, -2).contiguous()
            final_state = (
                StateTuple((state_spec.name,), (final,)) if return_final_state else None
            )
            return outputs, final_state
        if backward_path:
            dense_left = [item if item is not None else ones_row for item in left]
            dense_right = [item if item is not None else ones_row for item in right]
            dense_coeff = [
                item if item is not None else ones_scalar for item in coefficient
            ]
            outputs, final = executable.factored_recurrence_dense(
                operation_codes=tuple(codes),
                left_factors=dense_left,
                right_factors=dense_right,
                coefficients=dense_coeff,
                injection_left=injection_left,
                injection_right=injection_right,
                injection_scalars=injection_scalars,
                injection_orders=tuple(injection_orders),
                readout_axis=recipe.readout_axis,
                readouts=readouts,
                initial_state=initial_state,
                output_dtypes=tuple(dtypes),
            )
        else:
            outputs, final = factored_recurrence_chunked(
                operation_codes=tuple(codes),
                left_factors=left,
                right_factors=right,
                coefficients=coefficient,
                injection_left=injection_left,
                injection_right=injection_right,
                injection_scalars=injection_scalars,
                injection_orders=tuple(injection_orders),
                readout_axis=recipe.readout_axis,
                readouts=readouts,
                initial_state=initial_state,
                output_dtypes=tuple(dtypes),
                symmetric_rank_one=executable.symmetric_rank_one,
            )
        if recipe.transpose_state:
            final = final.transpose(-1, -2).contiguous()
        final_state = (
            StateTuple((state_spec.name,), (final,)) if return_final_state else None
        )
        return outputs, final_state
