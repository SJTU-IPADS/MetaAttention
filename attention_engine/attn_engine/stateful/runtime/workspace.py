"""Workspace contracts and the dense windows they permit.

Candidate compilation and tuning share one policy constructor. Both factored
scans and Scan Stage DAGs execute the resulting windows here, so output retention,
input slicing, and State continuation cannot drift between their adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import torch


FACTORED_INFERENCE_WINDOW = 1024


def _validate_workload(token_count, sequence_count, sequence_length):
    from ...stateful_operator import StatefulCompilationError

    for name, value in (
        ("token_count", token_count),
        ("sequence_count", sequence_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise StatefulCompilationError(
                "RUNTIME_RESOURCE", "workspace", {name: value}
            )
    if sequence_length is not None and (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < 0
        or token_count != sequence_count * sequence_length
    ):
        raise StatefulCompilationError(
            "RUNTIME_RESOURCE", "workspace", {"sequence_length": sequence_length}
        )


@dataclass(frozen=True)
class DenseWindowSchedule:
    window_tokens: int
    required_bytes: int

    def execute(
        self,
        inputs,
        sequence_axes: Mapping[str, int | None],
        runtime,
        state,
        execute_window: Callable,
        *,
        contiguous_inputs: bool = False,
    ):
        """Retain canonical B,H,T,... outputs and carry State between windows.

        Adapters return (outputs, State) and must request final State even when
        the caller does not export it. Input copies remain adapter-specific:
        Stage children require contiguous bindings, while factored preparation
        already normalizes strided windows.
        """
        assert self.window_tokens > 0
        length = runtime.uniform_length
        outputs = None
        for begin in range(0, length, self.window_tokens):
            count = min(self.window_tokens, length - begin)
            part = {}
            for name, axis in sequence_axes.items():
                value = inputs[name]
                if axis is not None:
                    value = value.narrow(axis, begin, count)
                    if contiguous_inputs:
                        value = value.contiguous()
                part[name] = value
            segment = replace(
                runtime,
                uniform_length=count,
                token_count=runtime.sequence_count * count,
            )
            segment_outputs, state = execute_window(part, state, segment)
            if outputs is None:
                outputs = tuple(
                    torch.empty(
                        (*value.shape[:2], length, *value.shape[3:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    for value in segment_outputs
                )
            for output, value in zip(outputs, segment_outputs, strict=True):
                output.narrow(2, begin, count).copy_(value)
            # No window output or contiguous input copy survives into the next
            # preparation; only the retained output and mathematical State do.
            del part, segment_outputs, value
        return outputs, state


@dataclass(frozen=True)
class WorkspacePolicy:
    """Per-token workspace bound, tightest where the compiled path allows.

    Aligned workloads may use a smaller per-token bound. Dense chunk scratch
    is charged per sequence and ceiling-divided chunk, including a final-State
    allowance. Windowed inference bounds scratch, not the complete retained output.
    """

    bytes_per_token: int
    maximum_bytes: int
    aligned_bytes_per_token: int | None = None
    alignment_tokens: int = 0
    bytes_per_chunk: int = 0
    chunk_tokens: int = 64
    inference_window_tokens: int = 0
    output_bytes_per_token: int = 0

    @classmethod
    def for_candidate(
        cls, candidate, estimate, *, compilation, algorithm, rank_one_factored
    ) -> WorkspacePolicy:
        """Bind identical accounting for the published and tuning executables."""
        factored = candidate.lowerer == "tilelang-factored-recurrence"
        windowed = (
            factored
            and compilation.mode == "dense"
            and not compilation.options.backward
        )
        output_bytes = 0
        if windowed:
            sizes = dict(compilation.role_sizes)
            state = algorithm.states[0]
            features = state.feature_roles
            if rank_one_factored.transpose_state:
                features = features[::-1]
            output_role = features[0 if rank_one_factored.readout_axis else 1]
            output_bytes = (
                4
                * sizes[state.head_role]
                * len(algorithm.readouts)
                * sizes[output_role]
            )
        return cls(
            bytes_per_token=estimate.workspace_bytes_per_token,
            maximum_bytes=compilation.target.workspace_bytes,
            aligned_bytes_per_token=estimate.aligned_workspace_bytes_per_token,
            alignment_tokens=estimate.alignment_tokens,
            bytes_per_chunk=estimate.workspace_bytes_per_chunk,
            # WY/short replay may use 16-token chunks even when the long
            # non-WY schedule uses 64; charge the conservative checkpoint count.
            chunk_tokens=16 if factored else 64,
            inference_window_tokens=FACTORED_INFERENCE_WINDOW if windowed else 0,
            output_bytes_per_token=output_bytes,
        )

    def required_bytes(
        self,
        token_count: int,
        *,
        sequence_count: int = 1,
        sequence_length: int | None = None,
    ) -> int:
        per_token = self.bytes_per_token
        if (
            self.aligned_bytes_per_token is not None
            and self.alignment_tokens
            and token_count % self.alignment_tokens == 0
        ):
            per_token = self.aligned_bytes_per_token
        length = sequence_length if sequence_length is not None else token_count
        live_tokens = token_count
        if self.inference_window_tokens:
            length = min(length, self.inference_window_tokens)
            live_tokens = min(token_count, sequence_count * length)
        required = live_tokens * per_token + token_count * self.output_bytes_per_token
        if (
            self.bytes_per_chunk
            and token_count
            and (
                self.aligned_bytes_per_token is None
                or per_token == self.aligned_bytes_per_token
            )
        ):
            chunks = (length + self.chunk_tokens - 1) // self.chunk_tokens
            required += sequence_count * (chunks + 1) * self.bytes_per_chunk
        if required > self.maximum_bytes:
            raise RuntimeError("runtime workload exceeds compiled resource envelope")
        return required

    def schedule(
        self, token_count, *, sequence_count=1, sequence_length=None
    ) -> DenseWindowSchedule:
        _validate_workload(token_count, sequence_count, sequence_length)
        required = self.required_bytes(
            token_count,
            sequence_count=sequence_count,
            sequence_length=sequence_length,
        )
        window = (
            min(sequence_length, self.inference_window_tokens)
            if self.inference_window_tokens and sequence_length is not None
            else 0
        )
        return DenseWindowSchedule(window, required)


@dataclass(frozen=True)
class StageWorkspace:
    """Conservative Stage trajectories plus child scans, optionally windowed.

    Child entries pair their workspace query with whether the affine prefix
    backward retains each doubling level. LogAddExp already charges its VJP.
    Checkpoint never discounts this non-checkpoint legality budget.
    """

    maximum_bytes: int
    materialized_bytes_per_token: int
    state_bytes_per_sequence: int
    output_bytes_per_token: int
    children: tuple[tuple[Callable[..., int], bool], ...]
    mode: str
    backward: bool
    windowable: bool

    def schedule(
        self, token_count, *, sequence_count=1, sequence_length=None
    ) -> DenseWindowSchedule:
        from ...stateful_operator import StatefulCompilationError

        _validate_workload(token_count, sequence_count, sequence_length)
        if (
            self.backward
            or self.mode != "dense"
            or sequence_length is None
            or not self.windowable
        ):
            return DenseWindowSchedule(
                0, self._required_bytes(token_count, sequence_count, sequence_length)
            )
        retained = token_count * self.output_bytes_per_token
        window = min(sequence_length, FACTORED_INFERENCE_WINDOW)
        while True:
            try:
                required = retained + self._required_bytes(
                    sequence_count * window, sequence_count, window
                )
                if required <= self.maximum_bytes:
                    return DenseWindowSchedule(window, required)
            except StatefulCompilationError as error:
                if error.category != "RUNTIME_RESOURCE":
                    raise
            if window <= 1:
                raise StatefulCompilationError(
                    "RUNTIME_RESOURCE",
                    "workspace",
                    {
                        "maximum": self.maximum_bytes,
                        "reason": "retained outputs and one-token stage scratch exceed budget",
                    },
                )
            window = max(1, window // 2)

    def _required_bytes(self, token_count, sequence_count, sequence_length):
        from ...stateful_operator import StatefulCompilationError

        required = (
            token_count * self.materialized_bytes_per_token
            + sequence_count * self.state_bytes_per_sequence
        )
        for workspace_bytes, affine_prefix in self.children:
            cost = workspace_bytes(
                token_count,
                sequence_count=sequence_count,
                sequence_length=sequence_length,
            )
            if self.backward and affine_prefix:
                cost *= 2 + max(1, (max(1, token_count) - 1).bit_length())
            required += cost
        if required > self.maximum_bytes:
            raise StatefulCompilationError(
                "RUNTIME_RESOURCE",
                "workspace",
                {"required": required, "maximum": self.maximum_bytes},
            )
        return required
