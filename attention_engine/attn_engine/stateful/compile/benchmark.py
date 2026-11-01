"""Compiler-owned workloads exercised through the ordinary compiled callable."""

import math

import torch


def invocation_factory(operator, compilation, binding, device):
    from ...stateful_operator import (
        Batch,
        Sequence,
        One,
        CompiledStatefulOperator,
        StateTuple,
    )

    roles = dict(compilation.role_sizes)
    states = tuple(operator.algorithm.states)

    def build(candidate, config, workload):
        from .tuning import workload_lengths

        executable = binding.bind(candidate, config)
        lengths = workload_lengths(workload)
        sizes = {**roles, Batch: len(lengths), Sequence: max(lengths), One: 1}
        shapes = {}
        for spec in operator.algorithm.inputs:
            if compilation.mode == "packed":
                shape = (
                    sum(lengths),
                    *(
                        roles[role] if role != One else 1
                        for role in spec.roles
                        if role not in (Batch, Sequence)
                    ),
                )
            else:
                shape = tuple(sizes[role] for role in spec.roles)
            shapes[spec.name] = shape
        # Conservative live-buffer bound before allocating or launching a candidate.
        elements = sum(math.prod(shape) for shape in shapes.values())
        workspace = executable.plan.workspace
        required = workspace.required_bytes(
            sum(lengths), sequence_count=len(lengths), sequence_length=max(lengths)
        )
        required += elements * 8
        required += (
            8
            * len(lengths)
            * sum(
                roles[spec.head_role]
                * math.prod(roles[role] for role in spec.feature_roles)
                for spec in states
            )
        )
        free, _ = torch.cuda.mem_get_info(device)
        if required > min(compilation.target.workspace_bytes, free // 2):
            raise RuntimeError("tuning workload exceeds safe resource envelope")
        inputs = {
            spec.name: torch.full(
                shapes[spec.name], 0.001, device=device, dtype=spec.dtype
            )
            for spec in operator.algorithm.inputs
        }
        initial_tensors = []
        for spec in states:
            feature_shape = tuple(roles[role] for role in spec.feature_roles)
            initial_tensors.append(
                torch.zeros(
                    len(lengths), roles[spec.head_role], *feature_shape, device=device
                )
            )
        initial = StateTuple(
            tuple(spec.name for spec in states), tuple(initial_tensors)
        )
        kwargs = {"initial_state": initial, "return_final_state": True}
        if compilation.mode == "packed":
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            kwargs["sequence_offsets"] = torch.tensor(
                offsets, dtype=torch.int32, device=device
            )
        compiled = CompiledStatefulOperator._from_executable(
            operator, compilation, executable
        )

        def invoke():
            with torch.no_grad():
                return compiled(**inputs, **kwargs)

        return invoke

    return build
