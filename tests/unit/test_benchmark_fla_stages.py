"""Registration, typed factories, recurrence and measurement-contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from testing import benchmark_fla_stages as bench

pytestmark = pytest.mark.unit


def test_registry_is_explicit_and_matrix_deterministic():
    assert set(bench.FAMILIES) == {
        "preconditioned-kda",
        "preconditioned-gdn",
        "lightnet",
    }
    rows = bench.matrix(tuple(bench.FAMILIES), (1, 64), (1, 2), 64, "bfloat16")
    assert len(rows) == 12
    assert rows[0] == ("preconditioned-kda", 1, 1)
    assert all(spec.chunk != spec.recurrent for spec in bench.FAMILIES.values())
    assert bench.FAMILIES["lightnet"].states == ("z", "memory")


@pytest.mark.parametrize(
    "args",
    [
        (["unknown"], [1], [1], 64, "float32"),
        (["lightnet"], [], [1], 64, "float32"),
        (["lightnet"], [1], [True], 64, "float32"),
        (["lightnet"], [1], [1], 0, "float32"),
        (["lightnet"], [1], [1], 64, "int8"),
    ],
)
def test_matrix_rejects_invalid_registration_types(args):
    with pytest.raises(ValueError):
        bench.matrix(*args)


@pytest.mark.parametrize("family", bench.FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_registered_public_factories_outputs_finals_and_vjp(monkeypatch, family, dtype):
    from attn_engine import Target

    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    torch.manual_seed(17)
    values, initial = bench.inputs_for(torch, family, 5, 2, 3, dtype, "cpu")
    operator, roles = bench.build_operator(family, 5, 2, 3, dtype)
    assert set(
        operator.algorithm.inputs[i].name for i in range(len(operator.algorithm.inputs))
    ) == set(values)
    assert {s.name for s in operator.algorithm.states} == set(
        bench.FAMILIES[family].states
    )
    assert values["query"].dtype == dtype
    assert all(t.dtype == torch.float32 for t in initial)
    compiled = operator.compile(
        role_sizes=roles, target=Target(), mode="dense", backward=True
    )
    call = bench.meta_call(compiled, family, values, initial)
    with torch.inference_mode():
        actual = call()
    expected = bench.oracle(torch, family, values, initial)
    assert bench.correctness(torch, actual, expected)["status"] == "VALID"
    assert bench.child_lowerers(compiled).keys() == compiled.children.keys()
    assert compiled.workspace_bytes(5, sequence_count=1, sequence_length=5) > 0
    values = {n: t.detach().requires_grad_() for n, t in values.items()}
    initial = tuple(t.detach().requires_grad_() for t in initial)
    tensors = (*values.values(), *initial)
    actual = bench.gradient_call(
        torch, bench.meta_call(compiled, family, values, initial), tensors
    )()
    expected = bench.gradient_call(
        torch, lambda: bench.oracle(torch, family, values, initial), tensors
    )()
    names = (*values, *bench.FAMILIES[family].states)
    assert (
        bench.correctness(
            torch, dict(zip(names, actual)), dict(zip(names, expected)), atol=0.02
        )["status"]
        == "VALID"
    )


def test_correctness_records_finals_and_baseline_difference():
    expected = {"output": torch.ones(2), "memory": torch.zeros(3), "atk": torch.ones(1)}
    bad = {**expected, "memory": torch.ones(3)}
    check = bench.correctness(torch, bad, expected)
    assert check["status"] == "INVALID"
    assert check["fields"]["memory"]["max_abs"] == 1
    assert (
        bench.correctness(torch, {"unused": None}, {"unused": None})["status"]
        == "VALID"
    )
    assert (
        bench.correctness(
            torch, {"x": torch.tensor(float("nan"))}, {"x": torch.tensor(0.0)}
        )["status"]
        == "INVALID"
    )


def test_measure_calls_only_warmed_public_closure_and_keeps_samples():
    calls, events = [], []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
            events.append(self)

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.0

    cuda = SimpleNamespace(
        synchronize=lambda: None,
        memory_allocated=lambda: 100,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 180,
        Event=Event,
    )
    fake = SimpleNamespace(cuda=cuda)
    result = bench.measure(fake, lambda: calls.append("public"), 2, 3)
    assert calls == ["public"] * 6  # warmup + one memory call + reps
    assert len(events) == 6
    assert result["cuda_samples_ms"] == [2.0] * 3
    assert result["public_cuda_event_ms"] == 2.0
    assert result["peak_allocated_bytes"] == 180
    assert result["peak_incremental_bytes"] == 80


def test_backward_is_separate_vjp_not_forward_plus_backward():
    calls = []
    x = torch.tensor(2.0, requires_grad=True)

    def forward():
        calls.append("forward")
        return {"output": x.square(), "final": x * 3}

    vjp = bench.gradient_call(torch, forward, (x,))
    assert calls == ["forward"]
    assert vjp()[0].item() == 7
    assert vjp()[0].item() == 7
    assert calls == ["forward"]
    assert "inference_mode" in bench.MEASUREMENT_CONTRACT["forward"]
    assert "layout copies excluded" in bench.MEASUREMENT_CONTRACT["preprocessing"]


def test_cli_contract_and_resource_classification():
    args = bench.parse_args(
        [
            "--family",
            "lightnet",
            "--heads",
            "2",
            "--dim",
            "32",
            "--dtype",
            "float32",
            "--workspace",
            "1024",
            "--reps",
            "2",
        ]
    )
    assert args.families == ["lightnet"] and args.workspace == 1024
    assert (
        bench._failure(RuntimeError("RUNTIME_RESOURCE workspace"), True)["status"]
        == "META_RUNTIME_REJECTED"
    )
    assert (
        bench._failure(RuntimeError("COMPILE_RESOURCE"), True)["status"]
        == "META_COMPILE_REJECTED"
    )


def test_backend_failure_does_not_skip_other_baselines(monkeypatch):
    monkeypatch.setattr(bench, "inputs_for", lambda *a: ({}, ()))

    class ContextTorch:
        manual_seed = staticmethod(lambda _: None)
        no_grad = staticmethod(lambda: torch.no_grad())
        inference_mode = staticmethod(lambda: torch.inference_mode())

    fake_torch = ContextTorch()
    monkeypatch.setattr(bench, "oracle", lambda *a: {})

    def rejected(*a):
        raise RuntimeError("COMPILE_RESOURCE")

    monkeypatch.setattr(bench, "build_operator", rejected)
    calls = []

    def unavailable(torch, family, variant, values, initial):
        calls.append(variant)
        raise ImportError("FLA unavailable")

    monkeypatch.setattr(bench, "fla_call", unavailable)
    result = bench.run_case(
        fake_torch, "lightnet", 1, 1, 64, "bfloat16", 1024, 1, 1, False
    )
    assert calls == ["chunk", "recurrent"]
    assert result["backends"]["meta_forward"]["status"] == "META_COMPILE_REJECTED"
    assert result["backends"]["fla_chunk_forward"]["error_type"] == "ImportError"
    assert result["backends"]["fla_recurrent_forward"]["status"] == "FAILED"


@pytest.mark.parametrize("family", bench.FAMILIES)
@pytest.mark.parametrize("variant", ["chunk", "recurrent"])
def test_fla_layout_binding_and_state_contract(monkeypatch, family, variant):
    values, initial = bench.inputs_for(torch, family, 2, 3, 4, torch.bfloat16, "cpu")
    spec = bench.FAMILIES[family]
    invocations = []

    def function(**kwargs):
        invocations.append(kwargs)
        assert kwargs["q"].shape == (1, 2, 3, 4)
        assert kwargs["q"].is_contiguous()
        assert kwargs["output_final_state"] is True
        o = torch.zeros_like(kwargs["v"])
        if spec.kind:
            assert kwargs["initial_state"] is initial[1]
            assert kwargs["initial_A_state"] is initial[0]
            if spec.kind == "gdn" or variant == "recurrent":
                assert kwargs["use_qk_l2norm_in_kernel"] is (spec.kind == "gdn")
            return o, initial[1], initial[0]
        assert kwargs["initial_state"][0] is initial[1]
        assert kwargs["initial_state"][1].shape == (1, 1, 3, 4)
        return o, (initial[1], initial[0].unsqueeze(1))

    module = SimpleNamespace(**{getattr(spec, variant): function})
    monkeypatch.setattr(bench.importlib, "import_module", lambda name: module)
    call = bench.fla_call(torch, family, variant, values, initial)
    assert not invocations  # prebinding is not execution
    result = call()
    assert result["output"].shape == values["value"].shape
    assert result[spec.states[0]] is not None
    assert result["memory"] is initial[1]
    native = {
        n: t.transpose(1, 2).contiguous().requires_grad_() for n, t in values.items()
    }
    bench.fla_call(torch, family, variant, native, initial, prebound=True)()
    assert invocations[-1]["q"] is native["query"]  # no layout graph inside VJP


@pytest.mark.gpu
@pytest.mark.parametrize("family", bench.FAMILIES)
@pytest.mark.parametrize("length", [1, 64, 1031])
def test_h20_public_inference_selected_lowerers_and_oracle(
    monkeypatch, gpu_device, family, length
):
    from attn_engine import Target

    monkeypatch.delenv("STATEFUL_OPERATOR_TEST_CPU", raising=False)
    torch.manual_seed(42)
    values, initial = bench.inputs_for(
        torch, family, length, 1, 64, torch.bfloat16, gpu_device
    )
    operator, roles = bench.build_operator(family, length, 1, 64, torch.bfloat16)
    compiled = operator.compile(
        role_sizes=roles,
        target=Target(workspace_bytes=8 << 30),
        mode="dense",
        backward=False,
    )
    lowerers = bench.child_lowerers(compiled)
    assert all(
        item["runtime_family"] != "stateful-reference" for item in lowerers.values()
    )
    if bench.FAMILIES[family].kind:
        assert (
            lowerers["preconditioner"]["selected_lowerer"]
            == "tilelang-chunked-stateful"
        )
        assert lowerers["memory"]["selected_lowerer"] == "tilelang-factored-recurrence"
    with torch.inference_mode():
        actual = bench.meta_call(compiled, family, values, initial)()
    expected = bench.oracle(
        torch,
        family,
        {n: t.cpu() for n, t in values.items()},
        tuple(t.cpu() for t in initial),
    )
    assert bench.correctness(torch, actual, expected)["status"] == "VALID"


def test_missing_requested_fla_source_is_failure_not_installed_fallback(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(bench, "FLA_SOURCE_ROOT", tmp_path / "missing")
    with pytest.raises(ImportError, match="source directory does not exist"):
        bench.fla_call(torch, "lightnet", "chunk", {}, ())


def test_run_case_enforces_inference_mode_for_public_forward(monkeypatch):
    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    original = bench.inputs_for
    monkeypatch.setattr(
        bench,
        "inputs_for",
        lambda torch, family, length, heads, dim, dtype, device: original(
            torch, family, length, heads, dim, dtype, "cpu"
        ),
    )
    seen = []

    def measure(torch, call, warmup, reps):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        call()
        seen.append("measured")
        return {}

    def fla(torch, family, variant, values, initial):
        def run():
            assert torch.is_inference_mode_enabled()
            return bench.oracle(torch, family, values, initial)

        return run

    monkeypatch.setattr(bench, "measure", measure)
    monkeypatch.setattr(bench, "fla_call", fla)
    record = bench.run_case(
        torch, "preconditioned-gdn", 2, 1, 3, torch.float32, 1 << 30, 0, 1, False
    )
    assert seen == ["measured"] * 3
    assert all(e["status"] == "OK" for e in record["backends"].values())


@pytest.mark.parametrize("timeout", [False, True])
def test_isolation_keeps_partial_record_and_records_child_failure(
    monkeypatch, tmp_path, timeout
):
    import json
    import subprocess
    from testing.benchmark_fla_stateful import GPUInfo

    monkeypatch.setattr(bench, "list_gpus", lambda: (GPUInfo(0, "H20", 4, 97871, 0),))

    def run(command, **kwargs):
        path = command[command.index("--output") + 1]
        with open(path, "w") as f:
            json.dump(
                {"family": "lightnet", "backends": {"meta_forward": {"status": "OK"}}},
                f,
            )
        if timeout:
            raise subprocess.TimeoutExpired(command, 1)
        return SimpleNamespace(returncode=-9)

    monkeypatch.setattr(bench.subprocess, "run", run)
    args = bench.parse_args(
        [
            "--family",
            "lightnet",
            "--seqlens",
            "1",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    record = bench.isolated_case(args, 0, "lightnet", 1, 1)
    assert record["status"] == ("CHILD_TIMEOUT" if timeout else "CHILD_FAILED")
    assert record["backends"]["meta_forward"]["status"] == "OK"
    assert record["backends"]["fla_chunk_backward"]["status"] == record["status"]
    assert len(record["backends"]) == 5


def test_isolation_rejects_busy_gpu_without_launch(monkeypatch):
    from testing.benchmark_fla_stateful import GPUInfo

    monkeypatch.setattr(
        bench, "list_gpus", lambda: (GPUInfo(0, "H20", 100, 97871, 0, (123,)),)
    )
    monkeypatch.setattr(
        bench.subprocess, "run", lambda *a, **k: pytest.fail("must not launch child")
    )
    record = bench.isolated_case(bench.parse_args([]), 0, "lightnet", 1, 1)
    assert record["status"] == "GPU_BUSY_REJECTED"
