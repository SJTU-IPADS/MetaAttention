"""Seeded, bounded mutations of a linear-attention Algorithm IR.

Levels measure authored edits, not algebraic equivalence or model identities.
Every family starts from the same scalar-decayed, single-write recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import random

import torch

from attn_engine import (
    AlgorithmIR,
    AxisScale,
    Batch,
    Exp,
    FeatureRole,
    HeadMapping,
    HeadRole,
    Input,
    ProductFactor,
    ProductInjection,
    RankOnePropagation,
    Sequence,
    Sigmoid,
    StateContraction,
    StateSpec,
    StateTransition,
    StatefulOperator,
    TensorInput,
)

FAMILIES = ("expression", "propagation", "rank", "write", "readout", "combined")
ROW = FeatureRole("mutation_row")
COLUMN = FeatureRole("mutation_column")
HEAD = HeadRole("mutation_heads")


@dataclass(frozen=True)
class Mutation:
    family: str
    level: int
    seed: int = 1701

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown mutation family {self.family!r}")
        if self.level < 0:
            raise ValueError("mutation level must be nonnegative")

    def amount(self, family: str) -> int:
        return self.level if self.family in (family, "combined") else 0


def _walk(value):
    if is_dataclass(value):
        yield value
        for field in fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def complexity(operator) -> dict[str, int]:
    """Count authored AST occurrences (shared nodes count at each use)."""
    from attn_engine.stateful_operator import InputExpression

    algorithm = operator.algorithm
    transition = algorithm.transitions[0]
    return {
        "expression_nodes": sum(
            isinstance(n, InputExpression) for n in _walk(algorithm)
        ),
        "propagations": len(transition.propagations),
        "rank_one_propagations": sum(
            isinstance(n, RankOnePropagation) for n in transition.propagations
        ),
        "writes": len(transition.injections),
        "readouts": len(algorithm.readouts),
        "inputs": len(algorithm.inputs),
    }


def make_operator(mutation: Mutation, *, heads=4, rows=64, columns=64):
    if min(heads, rows, columns) < 1:
        raise ValueError("head and feature sizes must be positive")
    rng = random.Random(mutation.seed)
    inputs = []

    def tensor(name, role=None, dtype=torch.bfloat16):
        roles = (Batch, HEAD, Sequence) + (() if role is None else (role,))
        inputs.append(TensorInput(name, roles, dtype))
        return Input(name)

    def transform(expression):
        # Each layer uses the preceding expression once, so authored size grows
        # linearly with level. Bounded inputs avoid exponential overflow.
        for _ in range(mutation.amount("expression")):
            motif = rng.randrange(3)
            if motif == 0:
                expression = expression * rng.uniform(0.8, 1.2) + 0.01
            elif motif == 1:
                expression = (Sigmoid(expression) - 0.5) * 4.0
            else:
                expression = (Exp(expression * 0.25) - 1.0) * 4.0
        return expression

    query = tensor("query", ROW)
    key = tensor("key", ROW)
    value = tensor("value", COLUMN)
    gate = tensor("gate", dtype=torch.float32)
    propagations = [AxisScale(Exp(gate), ())]
    for index in range(mutation.amount("propagation")):
        # Alternate axes with a seeded starting axis to exercise two-sided decay.
        role = (ROW, COLUMN)[(index + mutation.seed) % 2]
        factor = tensor(f"axis_gate_{index}", role, torch.float32)
        propagations.append(AxisScale(Exp(factor), (role,)))
    for index in range(mutation.amount("rank")):
        left = tensor(f"left_{index}", ROW)
        right = tensor(f"right_{index}", ROW)
        coefficient = tensor(f"coefficient_{index}", dtype=torch.float32)
        propagations.append(RankOnePropagation(ROW, left, right, coefficient))
    writes = 1 + mutation.amount("write")
    injections = []
    for index in range(writes):
        k = key if index == 0 else tensor(f"key_{index}", ROW)
        v = value if index == 0 else tensor(f"value_{index}", COLUMN)
        injections.append(
            ProductInjection(
                (
                    ProductFactor(transform(k) * (1.0 / writes), (ROW,)),
                    ProductFactor(transform(v), (COLUMN,)),
                )
            )
        )
    readouts = []
    for index in range(1 + mutation.amount("readout")):
        q = query if index == 0 else tensor(f"query_{index}", ROW)
        readouts.append(
            StateContraction(
                f"output_{index}",
                "memory",
                transform(q) * (rows**-0.5),
                ROW,
                torch.bfloat16,
            )
        )
    operator = StatefulOperator(
        AlgorithmIR(
            inputs=tuple(inputs),
            states=(StateSpec("memory", (Batch, HEAD, ROW, COLUMN)),),
            transition=StateTransition(
                "memory", propagations=tuple(propagations), injections=tuple(injections)
            ),
            readouts=tuple(readouts),
            head_mapping=HeadMapping({HEAD: HEAD}),
        )
    )
    return operator, {HEAD: heads, ROW: rows, COLUMN: columns}


def make_inputs(operator, sizes, *, batch, length, seed, device):
    """Generate on CPU for reproducibility across devices and profile prefixes."""
    import hashlib

    values = {}
    for spec in operator.algorithm.inputs:
        # Adding inputs must not change the shared control/variant input tensors.
        salt = int.from_bytes(hashlib.sha256(spec.name.encode()).digest()[:4], "little")
        generator = torch.Generator().manual_seed((seed + salt) % (2**63))
        shape = tuple(
            batch if role == Batch else length if role == Sequence else sizes[role]
            for role in spec.roles
        )
        value = torch.randn(shape, generator=generator)
        if "gate" in spec.name:
            value = -0.005 - value.abs() * 0.01
        elif spec.name.startswith("coefficient_"):
            value = -0.1 * value.sigmoid()
        elif spec.name.startswith(("left_", "right_")):
            value = torch.nn.functional.normalize(value, dim=-1) * 0.5
        else:
            value = value * 0.1
        values[spec.name] = value.to(device=device, dtype=spec.dtype)
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(
        batch, sizes[HEAD], sizes[ROW], sizes[COLUMN], generator=generator
    )
    return values, (initial * 0.01).to(device)


def oracle(operator, sizes, values, initial):
    # Reuse the independent benchmark token interpreter, never engine evaluators.
    from attn_engine import StateTuple
    from testing.benchmark_stateful_plans import _token_oracle

    return _token_oracle(
        operator, values, sizes, initial_state=StateTuple(("memory",), (initial,))
    )


def check_result(result, expected_outputs, expected_states):
    errors = {}
    pairs = [
        (f"output:{name}", result.outputs[name], value, 0.03, 0.001)
        for name, value in expected_outputs.items()
    ]
    pairs += [
        (f"state:{name}", result.final_state[name], value, 0.02, 0.001)
        for name, value in expected_states.items()
    ]
    for name, actual, expected, rtol, atol in pairs:
        actual = actual.detach().cpu().float()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        errors[name] = (actual - expected).abs().max().item()
    return errors
