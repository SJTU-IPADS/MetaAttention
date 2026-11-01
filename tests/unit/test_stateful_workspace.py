"""Resource contracts for the Stateful Operator plans.

`WorkspacePolicy` is the compiler's gate on sequence-scaled buffers, so the test
pins what each plan is allowed to charge per token and where the H20 envelope
stops accepting a workload. A plan that silently starts scaling with the token
count, or one that stops being reachable at long sequences, fails here.
"""

from __future__ import annotations

import pytest

from attn_engine import HeadRole, ProductFeatureRole, Target
from examples.gated_delta_rule import gated_delta_rule
from examples.gated_oja_rule import gated_oja_rule
from examples.linear_attention import gla_attention, simple_gla_attention

pytestmark = pytest.mark.unit

ENVELOPE = Target().workspace_bytes
HEADS = 8
ROWS = COLUMNS = 128


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


def _compile(factory, *, seqlen: int, backward: bool, size: int = ROWS, columns=None):
    kwargs = {"HQ": 1} if "gated_delta_rule" in factory.__module__ else {}
    operator = factory(1, HEADS, seqlen, size, columns or size, **kwargs)
    state = operator.algorithm.states[0]
    sizes: dict = {state.head_role: HEADS}
    sizes.update(zip(state.feature_roles, (size, columns or size), strict=True))
    for spec in operator.algorithm.inputs:
        for role in spec.roles:
            if isinstance(role, HeadRole):
                sizes.setdefault(role, HEADS)
            elif isinstance(role, ProductFeatureRole):
                sizes[role] = sizes[role.left] * sizes[role.right]
    return operator.compile(
        role_sizes=sizes,
        target=Target(),
        mode="dense",
        optimized=True,
        backward=backward,
    )


def _workspace(compiled):
    plan = compiled.operator._plans[compiled.specialization]
    assert plan.plan.candidate.lowerer is not None
    return plan.plan.candidate.lowerer, plan.plan.workspace


def test_factored_inference_bounds_scratch_but_still_charges_full_output():
    """Window scratch is bounded; retained output still scales with length."""
    lowerer, workspace = _workspace(
        _compile(gated_delta_rule, seqlen=2048, backward=False)
    )
    assert lowerer == "tilelang-factored-recurrence"
    assert 0 < workspace.required_bytes(2048) < ENVELOPE
    assert workspace.inference_window_tokens > 0
    short = workspace.required_bytes(2048)
    long = workspace.required_bytes(32768)
    assert long - short == (32768 - 2048) * workspace.output_bytes_per_token
    assert long < ENVELOPE
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(1 << 22)


def test_factored_short_replay_charges_every_checkpoint_before_rejecting():
    """A small workspace cap must not admit uncharged 16-token chunk buffers."""
    from dataclasses import replace

    from examples.rwkv7_state import rwkv7_state

    _, workspace = _workspace(_compile(rwkv7_state, seqlen=65, backward=False, size=64))
    length = 65
    token_bytes = length * (
        workspace.bytes_per_token + workspace.output_bytes_per_token
    )
    # Five chunk entries plus the final-State allowance, not two 64-token chunks.
    required = token_bytes + 6 * workspace.bytes_per_chunk
    bounded = replace(workspace, maximum_bytes=required)
    assert bounded.required_bytes(length, sequence_length=length) == required
    with pytest.raises(RuntimeError, match="resource envelope"):
        replace(workspace, maximum_bytes=required - 1).required_bytes(
            length, sequence_length=length
        )


def test_factored_window_workspace_counts_batch_tail_and_retained_output():
    from dataclasses import replace

    from attn_engine.stateful.compile.plan import WorkspacePolicy

    policy = WorkspacePolicy(
        bytes_per_token=2,
        maximum_bytes=1000,
        bytes_per_chunk=10,
        chunk_tokens=16,
        inference_window_tokens=64,
        output_bytes_per_token=4,
    )
    # Two full windows are live; output covers both entire 129-token sequences.
    required = 2 * 64 * 2 + 2 * (4 + 1) * 10 + 2 * 129 * 4
    boundary = replace(policy, maximum_bytes=required)
    assert (
        boundary.required_bytes(258, sequence_count=2, sequence_length=129) == required
    )
    with pytest.raises(RuntimeError, match="resource envelope"):
        replace(boundary, maximum_bytes=required - 1).required_bytes(
            258, sequence_count=2, sequence_length=129
        )
    assert policy.required_bytes(0, sequence_count=2, sequence_length=0) == 0
    assert policy.required_bytes(17, sequence_length=17) == 17 * 6 + 3 * 10


def test_factored_recurrence_charges_recorded_states_with_a_backward():
    """A compiled backward records the per-token State it replays from."""
    lowerer, workspace = _workspace(
        _compile(gated_delta_rule, seqlen=2048, backward=True)
    )
    assert lowerer == "tilelang-factored-recurrence"
    # one State per token per head, recorded in bf16
    assert workspace.bytes_per_token == HEADS * ROWS * COLUMNS * 2
    assert workspace.inference_window_tokens == 0
    assert workspace.output_bytes_per_token == 0
    assert workspace.required_bytes(2048) < ENVELOPE
    with pytest.raises(RuntimeError):
        workspace.required_bytes(32768)


def test_vector_plan_bounds_chunk_scratch_and_long_unaligned_sequences():
    """Aligned and unaligned lengths both charge bounded chunk scratch."""
    lowerer, workspace = _workspace(
        _compile(gla_attention, seqlen=2048, backward=False, size=64)
    )
    assert lowerer == "tilelang-vector-propagation"
    assert 0 < workspace.required_bytes(2048) < ENVELOPE
    assert 0 < workspace.required_bytes(32768) < ENVELOPE
    assert 0 < workspace.required_bytes(32769) < ENVELOPE


def test_fused_scalar_plan_charges_chunk_scratch_at_long_sequences():
    """The scalar schedule charges its bounded chunk carries and tail."""
    lowerer, workspace = _workspace(
        _compile(simple_gla_attention, seqlen=2048, backward=False, size=64)
    )
    assert lowerer == "tilelang-scalar-factorized"
    assert workspace.bytes_per_token == 0
    assert 0 < workspace.required_bytes(32768) < ENVELOPE


def test_chunk_workspace_counts_tail_and_batch_before_rejecting():
    from attn_engine.stateful.compile.plan import WorkspacePolicy

    workspace = WorkspacePolicy(
        bytes_per_token=2, maximum_bytes=500, bytes_per_chunk=100
    )
    # Two sequences of 32 tokens: each needs a carry and final-State allowance.
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(64, sequence_count=2, sequence_length=32)
    # One sequence uses fewer boundaries for the same token count.
    assert workspace.required_bytes(64, sequence_length=64) == 328
    # A tail adds a chunk, but the exact envelope edge is still accepted.
    workspace = WorkspacePolicy(
        bytes_per_token=2, maximum_bytes=430, bytes_per_chunk=100
    )
    assert workspace.required_bytes(65, sequence_length=65) == 430
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(66, sequence_length=66)
    assert workspace.required_bytes(0, sequence_count=2, sequence_length=0) == 0


def test_native_normalized_bounds_long_sequences_without_input_staging():
    from examples.normalized_linear_attention import normalized_linear_attention

    _, workspace = _workspace(
        _compile(normalized_linear_attention, seqlen=8192, backward=False, size=64)
    )
    # Two sequences and recursive carries fit without raising the production cap.
    assert (
        0
        < workspace.required_bytes(2 * 8192, sequence_count=2, sequence_length=8192)
        < ENVELOPE
    )
    # Batch still scales scratch; eliminating staging cannot bypass the gate.
    with pytest.raises(RuntimeError, match="resource envelope"):
        workspace.required_bytes(32 * 32769, sequence_count=32, sequence_length=32769)


@pytest.mark.parametrize("family", ["based", "rebased"])
@pytest.mark.parametrize("length", [64, 65, 130])
def test_product_workspace_rejects_uncharged_tail_and_batch_scratch(family, length):
    """Product carries cannot escape the cap through the non-product gate."""
    from dataclasses import replace

    from testing import benchmark_stateful_plans as benchmark

    case = replace(
        next(case for case in benchmark.CASES if case.name == family), batch=2, heads=8
    )
    operator = case.factory(**case.kwargs(length))
    sizes = benchmark.role_sizes_for(operator, case, length)
    compiled = operator.compile(
        role_sizes=sizes, target=Target(), mode="dense", optimized=True, backward=False
    )
    _, workspace = _workspace(compiled)
    # At least summary and exclusive carry must be charged; recursive scan
    # temporaries may make the conservative bound larger.
    state_elements = sum(
        sizes[state.feature_roles[0]]
        * (sizes[state.feature_roles[1]] if state.rank == 2 else 1)
        for state in operator.algorithm.states
    )
    chunk_bytes = 2 * 4 * case.heads * state_elements
    minimum = case.batch * length * workspace.bytes_per_token
    minimum += case.batch * ((length + 63) // 64 + 1) * chunk_bytes
    required = workspace.required_bytes(
        case.batch * length, sequence_count=case.batch, sequence_length=length
    )
    assert required >= minimum
    boundary = replace(workspace, maximum_bytes=required)
    assert (
        boundary.required_bytes(
            case.batch * length, sequence_count=case.batch, sequence_length=length
        )
        == required
    )
    with pytest.raises(RuntimeError, match="resource envelope"):
        replace(boundary, maximum_bytes=required - 1).required_bytes(
            case.batch * length, sequence_count=case.batch, sequence_length=length
        )


@pytest.mark.parametrize(
    "factory", [gated_delta_rule, gated_oja_rule, gla_attention, simple_gla_attention]
)
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("length", [65, 1031])
def test_tuning_invocation_uses_the_published_workspace_contract(
    monkeypatch, factory, backward, length
):
    import torch

    from attn_engine.stateful.compile.benchmark import invocation_factory
    from attn_engine.stateful.compile.binding import CandidateBinding
    from attn_engine.stateful.compile.plan import WorkloadPoint
    from attn_engine.stateful_operator import CompiledStatefulOperator

    bindings = []
    bind = CandidateBinding.bind

    def record_binding(self, candidate, config):
        bindings.append(self)
        return bind(self, candidate, config)

    monkeypatch.setattr(CandidateBinding, "bind", record_binding)
    compiled = _compile(
        factory,
        seqlen=length,
        backward=backward,
        size=64,
        columns=128 if factory is gated_oja_rule else 64,
    )
    executable = compiled.operator._plans[compiled.specialization]
    recorded = []

    def record_invocation(probe, **kwargs):
        recorded.append(probe._executable.plan.workspace)

    monkeypatch.setattr(CompiledStatefulOperator, "__call__", record_invocation)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 << 40, 1 << 40))
    build = invocation_factory(
        compiled.operator,
        compiled.compilation,
        bindings[0],
        torch.device("cpu"),
    )
    invoke = build(
        executable.plan.candidate,
        executable.plan.kernel_config.values,
        WorkloadPoint(2, 2 * length, "uniform"),
    )
    invoke()
    assert recorded == [executable.plan.workspace]


@pytest.mark.parametrize("backward", [False, True])
def test_compiled_workspace_query_matches_runtime_resource_gate(backward):
    from attn_engine import StatefulCompilationError

    compiled = _compile(gated_delta_rule, seqlen=1031, backward=backward, size=64)
    _, policy = _workspace(compiled)
    assert compiled.workspace_bytes(
        2062, sequence_count=2, sequence_length=1031
    ) == policy.required_bytes(2062, sequence_count=2, sequence_length=1031)
    with pytest.raises(StatefulCompilationError) as error:
        compiled.workspace_bytes(1 << 25, sequence_length=1 << 25)
    assert error.value.category == "RUNTIME_RESOURCE"
    assert error.value.path == "workspace"


@pytest.mark.parametrize(
    "workload",
    [
        {"token_count": -1},
        {"token_count": True},
        {"token_count": 2.5},
        {"token_count": 2, "sequence_count": -1},
        {"token_count": 2, "sequence_length": False},
        {"token_count": 2, "sequence_length": -1},
        {"token_count": 2, "sequence_count": 2, "sequence_length": 2},
    ],
)
def test_compiled_workspace_query_rejects_malformed_workloads(workload):
    from attn_engine import StatefulCompilationError

    compiled = _compile(gated_delta_rule, seqlen=65, backward=False, size=64)
    with pytest.raises(StatefulCompilationError) as error:
        compiled.workspace_bytes(**workload)
    assert error.value.category == "RUNTIME_RESOURCE"
    assert error.value.path == "workspace"
