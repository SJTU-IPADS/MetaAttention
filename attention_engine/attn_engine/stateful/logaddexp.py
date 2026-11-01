"""A closed LogAddExp prefix algebra, not a general State-feedback language.

The only transition is z' = logaddexp(z, logits). Its associative summary is
one vector and its identity is -inf. Readouts follow FLA's LightNet gate
contract: exp(logits-z') and z-z', with a zero gate for an empty prior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from ..stateful_operator import (
    Batch,
    CompileOptions,
    ExecutionResult,
    FeatureRole,
    HeadMapping,
    HeadRole,
    NamedOutputTuple,
    Sequence,
    StateSpec,
    StateTuple,
    StatefulCompilationError,
    SCHEMA_VERSION,
    _Compilation,
    TensorInput,
    _canonical_identity,
)
from .runtime.validation import (
    compile_validation,
    validate_call,
    validate_compile_roles,
)


class _LogAddExp(torch.autograd.Function):
    """The identity/identity VJP is zero rather than torch's NaN."""

    @staticmethod
    def forward(ctx, left, right):
        result = torch.logaddexp(left, right)
        ctx.save_for_backward(left, right, result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        left, right, result = ctx.saved_tensors
        finite = torch.isfinite(result)
        base = torch.where(finite, result, 0.0)
        dl = torch.where(finite, torch.exp(left - base), 0.0)
        dr = torch.where(finite, torch.exp(right - base), 0.0)
        return (gradient * dl).sum_to_size(left.shape), (gradient * dr).sum_to_size(
            right.shape
        )


def logaddexp_combine(left, right):
    """Associative finite-or-negative-infinity summary composition."""
    return _LogAddExp.apply(left, right)


@dataclass(frozen=True)
class _LogReadout:
    name: str
    output_dtype: torch.dtype = torch.float32


@dataclass(frozen=True, init=False)
class LogAddExpScanIR:
    """Typed vector-logit scan with one FP32 vector State and two readouts.

    All physical input role permutations are supported. Output layout is
    canonical B,H,T,K (packed T,H,K); State layout is B,H,K. -inf logits are
    permitted as masked identity elements, while NaN/+inf are outside this
    algebra. Neither arbitrary transitions nor feedback expressions exist.
    """

    inputs: tuple[TensorInput, ...]
    states: tuple[StateSpec, ...]
    readouts: tuple[_LogReadout, ...]
    head_mapping: HeadMapping

    def __init__(self, logits: TensorInput, state: StateSpec | None = None):
        path = "logaddexp.inputs"
        if not isinstance(logits, TensorInput):
            raise StatefulCompilationError("IR_SCHEMA", path, {})
        heads = [r for r in logits.roles if isinstance(r, HeadRole)]
        features = [r for r in logits.roles if isinstance(r, FeatureRole)]
        if (
            len(logits.roles) != 4
            or Batch not in logits.roles
            or Sequence not in logits.roles
            or len(heads) != 1
            or len(features) != 1
        ):
            raise StatefulCompilationError(
                "IR_TYPE", path, {"expected": "Batch, Head, Sequence, Feature"}
            )
        if state is None:
            state = StateSpec("z", (Batch, heads[0], features[0]))
        if (
            not isinstance(state, StateSpec)
            or state.roles != (Batch, heads[0], features[0])
            or not state.name.isidentifier()
        ):
            raise StatefulCompilationError(
                "IR_TYPE",
                "logaddexp.states",
                {"expected": "vector with matching Head and Feature"},
            )
        object.__setattr__(self, "inputs", (logits,))
        object.__setattr__(self, "states", (state,))
        object.__setattr__(
            self, "readouts", (_LogReadout("normalized_key"), _LogReadout("log_gate"))
        )
        object.__setattr__(self, "head_mapping", HeadMapping({heads[0]: heads[0]}))

    @property
    def _structural_identity(self):
        return _canonical_identity(self)

    def output_spec(self, name):
        if name not in tuple(r.name for r in self.readouts):
            raise StatefulCompilationError(
                "IR_TYPE", "logaddexp.outputs", {"unknown": name}
            )
        state = self.states[0]
        return TensorInput(
            name,
            (Batch, state.head_role, Sequence, *state.feature_roles),
            torch.float32,
        )


class LogAddExpOperator:
    """Metadata-only compiler sharing the public Stateful validation contract."""

    def __init__(self, algorithm):
        if not isinstance(algorithm, LogAddExpScanIR):
            raise StatefulCompilationError("IR_SCHEMA", "algorithm", {})
        self.algorithm = algorithm

    def __call__(self, **kwargs):
        raise RuntimeError(
            "StatefulOperator must be explicitly compiled before invocation; "
            "call operator.compile(...) and invoke the returned handle"
        )

    def compile(
        self,
        *,
        role_sizes,
        target,
        mode,
        optimized=True,
        autotune=False,
        backward=False,
    ):
        if any(not isinstance(v, bool) for v in (optimized, autotune, backward)):
            raise StatefulCompilationError("COMPILE_OPTIONS", "options", {})
        ordered = validate_compile_roles(self.algorithm, role_sizes, target, mode)
        compilation = _Compilation(
            ordered,
            target,
            mode,
            CompileOptions(optimized, autotune, backward),
            compile_validation(
                self.algorithm,
                reference_required=True,
                state_identity=-torch.inf,
                allow_empty_dense=True,
            ),
        )
        if (
            target.device_type != "cuda"
            or not isinstance(target.product_family, str)
            or target.product_family.upper() != "H20"
            or target.capability != (9, 0)
            or not isinstance(target.workspace_bytes, int)
            or isinstance(target.workspace_bytes, bool)
            or target.workspace_bytes < 0
        ):
            raise StatefulCompilationError(
                "COMPILE_TARGET",
                "target",
                {"expected": "CUDA H20 and nonnegative workspace budget"},
            )
        # Keep the existing specialization identity independent of validation metadata.
        identity = hashlib.sha256(
            json.dumps(
                (SCHEMA_VERSION, self.algorithm._structural_identity), sort_keys=True
            ).encode()
        ).hexdigest()
        specialization = _canonical_identity(
            (identity, compilation.role_sizes, target, mode, compilation.options)
        )
        return CompiledLogAddExp(self, compilation, specialization)


def _segmented_prefix(values, ids):
    """Hillis-Steele doubling: O(log T) depth, no per-sequence host loop."""
    distance = 1
    while distance < values.shape[0]:
        same = (ids[distance:] == ids[:-distance])[:, None, None]
        left = torch.where(same, values[:-distance], -torch.inf)
        values = torch.cat(
            (values[:distance], logaddexp_combine(left, values[distance:])), dim=0
        )
        distance *= 2
    return values


class CompiledLogAddExp:
    runtime_family = "segmented-logaddexp"

    def __init__(self, operator, compilation, specialization):
        self.operator, self.compilation, self.specialization = (
            operator,
            compilation,
            specialization,
        )

    def workspace_bytes(self, token_count, *, sequence_count=1, sequence_length=None):
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
        sizes = dict(self.compilation.role_sizes)
        state = self.operator.algorithm.states[0]
        width = sizes[state.head_role] * sizes[state.feature_roles[0]]
        levels = max(1, (max(1, token_count) - 1).bit_length())
        # Conservatively cover copies, masks, saved combine operands and VJP
        # intermediates at every doubling level, plus State and index vectors.
        factor = 32 + (48 * levels if self.compilation.options.backward else 16)
        required = (
            width * (token_count * factor + sequence_count * 32)
            + (token_count + sequence_count + 1) * 32
        )
        if required > self.compilation.target.workspace_bytes:
            raise StatefulCompilationError(
                "RUNTIME_RESOURCE",
                "workspace",
                {
                    "required": required,
                    "maximum": self.compilation.target.workspace_bytes,
                },
            )
        return required

    def __call__(
        self,
        *,
        initial_state=None,
        sequence_offsets=None,
        return_final_state=False,
        **inputs,
    ):
        c, op = self.compilation, self.operator
        runtime, sizes, states, _ = validate_call(
            inputs,
            initial_state=initial_state,
            sequence_offsets=sequence_offsets,
            role_sizes=c.role_sizes,
            validation=c.validation,
            mode=c.mode,
            target=c.target,
            backward=c.options.backward,
        )
        state = states[op.algorithm.states[0].name]
        self.workspace_bytes(
            runtime.token_count,
            sequence_count=runtime.sequence_count,
            sequence_length=runtime.uniform_length,
        )
        spec = op.algorithm.inputs[0]
        x = inputs[spec.name]
        roles = (
            spec.roles
            if not runtime.packed
            else (Sequence, *tuple(r for r in spec.roles if r not in (Batch, Sequence)))
        )
        canonical = (
            Sequence,
            op.algorithm.states[0].head_role,
            op.algorithm.states[0].feature_roles[0],
        )
        if runtime.packed:
            x = x.permute(*(roles.index(r) for r in canonical)).float()
            offsets = sequence_offsets.long()
        else:
            x = (
                x.permute(*(roles.index(r) for r in (Batch, *canonical)))
                .reshape(
                    runtime.token_count, runtime.state_heads, *runtime.feature_sizes
                )
                .float()
            )
            offsets = (
                torch.arange(
                    runtime.sequence_count + 1, device=x.device, dtype=torch.long
                )
                * runtime.uniform_length
            )
        for value in (x, state):
            torch._assert_async(
                torch.all(torch.isfinite(value) | torch.isneginf(value)),
                "LogAddExp accepts finite values and -inf only",
            )
        tokens = torch.arange(runtime.token_count, device=x.device)
        ids = torch.bucketize(tokens, offsets[1:], right=True)
        z = logaddexp_combine(_segmented_prefix(x, ids), state[ids])
        extended = torch.cat((state, z), dim=0)
        first = tokens == offsets[ids]
        prior_index = torch.where(first, ids, runtime.sequence_count + tokens - 1)
        prior = extended[prior_index]
        finite = torch.isfinite(z)
        base = torch.where(finite, z, 0.0)
        key = torch.where(finite, torch.exp(x - base), 0.0)
        has_prior = torch.isfinite(prior)
        gate = torch.where(has_prior, torch.where(has_prior, prior, 0.0) - base, 0.0)
        final_index = torch.where(
            offsets[1:] > offsets[:-1],
            runtime.sequence_count + offsets[1:] - 1,
            torch.arange(runtime.sequence_count, device=x.device),
        )
        final = extended[final_index]
        if not runtime.packed:
            shape = (
                runtime.sequence_count,
                runtime.uniform_length,
                runtime.state_heads,
                *runtime.feature_sizes,
            )
            key, gate = (
                v.reshape(shape).permute(0, 2, 1, 3).contiguous() for v in (key, gate)
            )
        return ExecutionResult(
            NamedOutputTuple(c.validation.output_names, (key, gate)),
            StateTuple(c.validation.state_names, (final.contiguous(),))
            if return_final_state
            else None,
        )
