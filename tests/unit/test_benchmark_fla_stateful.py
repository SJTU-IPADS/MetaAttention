from __future__ import annotations

import pytest

from testing.benchmark_fla_stateful import (
    P0_FAMILIES,
    P0_HEADS,
    P0_SEQLENS,
    benchmark_matrix,
    choose_idle_gpu,
    GPUInfo,
)

pytestmark = pytest.mark.unit


def test_first_phase_matrix_is_deterministic_and_complete():
    matrix = benchmark_matrix(
        families=("linear", "gla"),
        batches=(1, 2),
        heads=(1, 8),
        seqlens=(63, 64),
    )
    assert len(matrix) == 2 * 2 * 2 * 2
    assert matrix == benchmark_matrix(
        families=("linear", "gla"),
        batches=(1, 2),
        heads=(1, 8),
        seqlens=(63, 64),
    )
    assert matrix[0] == ("linear", 1, 1, 63, "output-only")


def test_first_phase_defaults_only_cover_supported_families():
    matrix = benchmark_matrix()
    assert {row[0] for row in matrix} == set(P0_FAMILIES)
    assert {row[2] for row in matrix} == set(P0_HEADS)
    assert {row[3] for row in matrix} == set(P0_SEQLENS)


def test_matrix_rejects_unsupported_or_invalid_axes():
    with pytest.raises(ValueError, match="linear"):
        benchmark_matrix(families=("unknown",))
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark_matrix(seqlens=())
    with pytest.raises(ValueError, match="must be positive"):
        benchmark_matrix(heads=(0,))
    with pytest.raises(ValueError, match="continuation"):
        benchmark_matrix(seqlens=(1,), contracts=("continuation",))


def test_extended_family_defaults_are_registered():
    assert set(P0_FAMILIES) == {
        "linear",
        "normalized-linear",
        "simple-gla",
        "gla",
        "retention",
        "retnet",
        "rwkv6",
        "rwkv7",
        "gdn",
        "gsa",
        "hgrn",
        "based",
        "rebased",
        "delta-rule",
        "kda",
        "gdn2",
        "comba",
        "iplr",
        "dplr",
        "oja",
        "delta-product",
        "abc",
        "preconditioned-kda",
        "preconditioned-gdn",
        "lightnet",
    }


def test_idle_gpu_selection_rejects_busy_and_selects_most_free():
    gpus = (
        GPUInfo(0, "busy", 100, 1000, 10, (123,)),
        GPUInfo(1, "idle-small", 100, 9000, 0),
        GPUInfo(2, "idle-large", 100, 12000, 0),
    )
    assert choose_idle_gpu(gpus).index == 2
    with pytest.raises(RuntimeError, match="no GPU"):
        choose_idle_gpu((gpus[0],))


def test_rwkv7_adapter_maps_decay_before_removal():
    import torch
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "rwkv7")
    operator = case.factory(**case.kwargs(2))
    inputs = benchmark.inputs_for(operator, case, 2, torch.device("cpu"))
    adapter = _fla_call(case, inputs)
    assert adapter is not None
    assert adapter.comparable is True
    values = adapter.prepare(inputs)
    native = {name: value.transpose(1, 2) for name, value in inputs.items()}
    torch.testing.assert_close(
        values["a"], -native["removal"].float() * native["decay"].exp()
    )
    torch.testing.assert_close(
        values["b"], native["key"].float() * native["beta"].unsqueeze(-1)
    )
    torch.testing.assert_close(
        values["value"], native["value"].float() * native["beta"].unsqueeze(-1)
    )


def test_gdn_adapter_passes_gate_and_beta_by_name(monkeypatch):
    import torch
    import fla.ops.gated_delta_rule as fla_gdn
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "gdn")
    operator = case.factory(**case.kwargs(2))
    inputs = benchmark.inputs_for(operator, case, 2, torch.device("cpu"))

    def recurrent(q, k, v, g=None, gk=None, gv=None, beta=None, **kwargs):
        assert gk is None and gv is None
        torch.testing.assert_close(g, inputs["gate"].transpose(1, 2))
        torch.testing.assert_close(beta, inputs["beta"].transpose(1, 2))
        return v, None

    monkeypatch.setattr(fla_gdn, "fused_recurrent_gated_delta_rule", recurrent)
    adapter = _fla_call(case, inputs)
    adapter.invoke(adapter.prepare(inputs), None, False)


@pytest.mark.parametrize(
    "family,module_name,function_name,mapping",
    (
        ("delta-rule", "delta_rule", "fused_recurrent_delta_rule", {"beta": "beta"}),
        ("kda", "kda", "fused_recurrent_kda", {"g": "gate", "beta": "beta"}),
        (
            "gdn2",
            "gdn2",
            "fused_recurrent_gdn2",
            {"g": "gate", "b": "erase", "w": "write"},
        ),
        (
            "comba",
            "comba",
            "fused_recurrent_comba",
            {"p": "prediction_key", "g": "gate", "beta": "beta"},
        ),
        (
            "iplr",
            "generalized_delta_rule.iplr",
            "fused_recurrent_iplr_delta_rule",
            {"a": "alpha", "b": "beta"},
        ),
        (
            "dplr",
            "generalized_delta_rule.dplr",
            "fused_recurrent_dplr_delta_rule",
            {"a": "alpha", "b": "beta", "gk": "gate"},
        ),
        (
            "oja",
            "gated_oja_rule",
            "fused_recurrent_gated_oja_rule",
            {"gv": "gate", "beta": "beta"},
        ),
    ),
)
def test_new_adapters_preserve_named_arguments_and_continuation(
    monkeypatch, family, module_name, function_name, mapping
):
    import sys
    from types import SimpleNamespace
    import torch
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == family)
    operator = case.factory(**case.kwargs(3))
    inputs = benchmark.inputs_for(operator, case, 3, torch.device("cpu"))
    initial = torch.randn(case.batch, case.heads, case.dim, case.dim_value)

    def recurrent(*args, **kwargs):
        assert kwargs["initial_state"] is initial
        assert kwargs["output_final_state"] is True
        assert kwargs["scale"] == case.dim**-0.5
        for argument, input_name in mapping.items():
            expected = inputs[input_name].transpose(1, 2)
            if family == "comba" and argument == "p":
                expected = expected.float() * inputs["gate"].transpose(
                    1, 2
                ).exp().unsqueeze(-1)
            torch.testing.assert_close(kwargs[argument], expected)
        value = kwargs["v"] if family == "comba" else args[2]
        return value, initial

    monkeypatch.setitem(
        sys.modules,
        f"fla.ops.{module_name}",
        SimpleNamespace(**{function_name: recurrent}),
    )
    adapter = _fla_call(case, inputs)
    outputs, states = adapter.split(
        adapter.invoke(adapter.prepare(inputs), initial, True)
    )
    torch.testing.assert_close(outputs["output"], inputs["value"])
    assert states["memory"] is initial


def test_delta_product_adapter_interleaves_writes_inside_each_token(monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "delta-product")
    operator = case.factory(**case.kwargs(3))
    inputs = benchmark.inputs_for(operator, case, 3, torch.device("cpu"))

    def chunk(q, k, v, **kwargs):
        assert kwargs["num_householder"] == 2
        for actual, name in ((k, "key"), (v, "value"), (kwargs["beta"], "beta")):
            for step in range(2):
                torch.testing.assert_close(
                    actual[:, step::2], inputs[f"{name}_{step}"].transpose(1, 2)
                )
        torch.testing.assert_close(kwargs["g"], inputs["gate"].transpose(1, 2))
        return v[:, ::2], None

    monkeypatch.setitem(
        sys.modules,
        "fla.ops.gated_delta_product",
        SimpleNamespace(chunk_gated_delta_product=chunk),
    )
    adapter = _fla_call(case, inputs)
    adapter.invoke(adapter.prepare(inputs), None, False)


def test_abc_adapter_limits_comparison_to_output_only(monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from testing.benchmark_fla_stateful import _fla_call
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == "abc")
    operator = case.factory(**case.kwargs(3))
    inputs = benchmark.inputs_for(operator, case, 3, torch.device("cpu"))

    def chunk(q, k, v, s, **kwargs):
        assert s.dtype == inputs["query"].dtype
        torch.testing.assert_close(s.float(), inputs["slot_logits"].transpose(1, 2))
        assert kwargs == {"initial_state": None, "output_final_state": False}
        return v, None

    monkeypatch.setitem(sys.modules, "fla.ops.abc", SimpleNamespace(chunk_abc=chunk))
    adapter = _fla_call(case, inputs)
    assert not adapter.state_contract
    assert "three raw accumulators" in adapter.note
    adapter.invoke(adapter.prepare(inputs), None, False)


def test_meta_failure_status_distinguishes_runtime_resource():
    from testing.benchmark_fla_stateful import _meta_failure_status

    assert (
        _meta_failure_status(RuntimeError("RUNTIME_RESOURCE at workspace"))
        == "META_RUNTIME_REJECTED"
    )
    assert (
        _meta_failure_status(RuntimeError("COMPILE_RESOURCE at workspace"))
        == "META_COMPILE_REJECTED"
    )
    assert _meta_failure_status(ValueError("bad input")) == "META_FAILED"


def test_valid_pair_requires_all_correctness_and_statuses():
    from testing.benchmark_fla_stateful import _valid_pair

    record = {
        "status": "OK",
        "fla_status": "OK",
        "fla_contract_comparable": True,
        "meta_correctness": {"status": "VALID"},
        "fla_correctness": {"status": "VALID"},
        "pair_correctness": {"status": "VALID"},
    }
    assert _valid_pair(record)
    for field in ("meta_correctness", "fla_correctness", "pair_correctness"):
        invalid = {**record, field: {"status": "INVALID"}}
        assert not _valid_pair(invalid)
    assert not _valid_pair({**record, "status": "OK", "fla_status": "FAILED"})


def test_csv_rows_hide_measurements_for_invalid_pairs():
    from testing.benchmark_fla_stateful import _csv_rows

    record = {
        "schema_version": 5,
        "family": "linear",
        "batch": 1,
        "heads": 1,
        "seqlen": 3,
        "contract": "output-only",
        "status": "OK",
        "fla_status": "FAILED",
        "meta": {"public_cuda_event_ms": 1.0},
        "fla": {"public_cuda_event_ms": 2.0},
        "speedup_public_cuda_event_ms": 2.0,
    }
    row = _csv_rows([record])[0]
    assert row["meta_ms"] is None
    assert row["fla_ms"] is None
    assert row["speedup"] is None


@pytest.mark.gpu
def test_retention_long_final_state_and_continuation(gpu_device):
    import torch
    from dataclasses import replace
    from testing import benchmark_stateful_plans as benchmark

    torch.set_num_threads(1)
    torch.manual_seed(42)
    case = replace(
        next(item for item in benchmark.CASES if item.name == "retention"),
        batch=2,
        heads=16,
    )
    operator = case.factory(**case.kwargs(8192))
    sizes = benchmark.role_sizes_for(operator, case, 8192)
    compiled = operator.compile(
        role_sizes=sizes, target=benchmark.Target(), mode="dense", optimized=True
    )
    inputs = benchmark.inputs_for(operator, case, 8192, gpu_device)
    expected, expected_states = benchmark._token_oracle(operator, inputs, sizes)
    for contract in ("final-state", "continuation"):
        parts = (
            benchmark._split_inputs(operator, inputs, 4096)
            if contract == "continuation"
            else [inputs]
        )
        outputs, states = benchmark._meta_call(compiled, parts, contract)()
        correctness = benchmark._correctness(
            outputs, states, expected, expected_states, contract
        )
        assert correctness["status"] == "VALID", correctness


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["linear", "retention", "retnet"])
def test_long_scalar_families_use_safe_schedule(gpu_device, family):
    import torch
    from testing import benchmark_stateful_plans as benchmark

    case = next(item for item in benchmark.CASES if item.name == family)
    case = case.__class__(**{**case.__dict__, "batch": 2, "heads": 16})
    operator = case.factory(**case.kwargs(8192))
    sizes = benchmark.role_sizes_for(operator, case, 8192)
    compiled = operator.compile(
        role_sizes=sizes, target=benchmark.Target(), mode="dense", optimized=True
    )
    inputs = benchmark.inputs_for(operator, case, 8192, gpu_device)
    actual, _ = benchmark._meta_call(compiled, [inputs], "output-only")()
    expected, _ = benchmark._token_oracle(operator, inputs, sizes)
    assert torch.allclose(
        actual["output"].cpu().float(), expected["output"], rtol=0.02, atol=0.05
    )


@pytest.mark.parametrize("baseline", ["recurrent", "prefill"])
@pytest.mark.parametrize("family", P0_FAMILIES)
def test_explicit_baseline_selects_public_api_without_fallback(
    monkeypatch, family, baseline
):
    import sys
    from types import SimpleNamespace
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark

    registry = (
        runner.DELTA_BASELINE_APIS
        if family in runner.DELTA_BASELINE_APIS
        else runner.BASELINE_APIS
    )
    module, recurrent, prefill = registry[family]
    function_name = recurrent if baseline == "recurrent" else prefill

    def function(*args, **kwargs):
        raise AssertionError("adapter construction must not execute FLA")

    monkeypatch.setitem(
        sys.modules,
        f"fla.ops.{module}",
        SimpleNamespace(**{name: function for name in (recurrent, prefill) if name}),
    )
    case = next(item for item in benchmark.CASES if item.name == family)
    if function_name is None:
        with pytest.raises(runner.BaselineUnavailableError, match=f"{baseline} API"):
            runner._fla_call(case, {}, baseline)
    else:
        adapter = runner._fla_call(case, {}, baseline)
        assert adapter.function_name == f"fla.ops.{module}.{function_name}"


def test_missing_prefill_function_does_not_use_recurrent(monkeypatch):
    import sys
    from types import SimpleNamespace
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark

    monkeypatch.setitem(
        sys.modules, "fla.ops.kda", SimpleNamespace(fused_recurrent_kda=lambda: None)
    )
    case = next(item for item in benchmark.CASES if item.name == "kda")
    with pytest.raises(runner.BaselineUnavailableError, match="chunk_kda"):
        runner._fla_call(case, {}, "prefill")


@pytest.mark.parametrize("baseline", ["recurrent", "prefill"])
def test_comba_prediction_conversion_depends_on_baseline(monkeypatch, baseline):
    import sys
    from types import SimpleNamespace
    import torch
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark

    monkeypatch.setitem(
        sys.modules,
        "fla.ops.comba",
        SimpleNamespace(fused_recurrent_comba=lambda: None, chunk_comba=lambda: None),
    )
    case = next(item for item in benchmark.CASES if item.name == "comba")
    operator = case.factory(**case.kwargs(3))
    inputs = benchmark.inputs_for(operator, case, 3, torch.device("cpu"))
    values = runner._fla_call(case, inputs, baseline).prepare(inputs)
    prediction = inputs["prediction_key"].transpose(1, 2)
    expected = (
        prediction.float() * inputs["gate"].transpose(1, 2).exp().unsqueeze(-1)
        if baseline == "recurrent"
        else prediction
    )
    torch.testing.assert_close(values["prediction_key"], expected)


@pytest.mark.parametrize("family", ["linear", "simple-gla", "gla", "rwkv7"])
def test_prefill_preserves_authored_transforms_before_fp32_promotion(
    monkeypatch, family
):
    import sys
    from types import SimpleNamespace
    import torch
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark

    module, recurrent, prefill = runner.BASELINE_APIS[family]
    monkeypatch.setitem(
        sys.modules,
        f"fla.ops.{module}",
        SimpleNamespace(**{recurrent: lambda: None, prefill: lambda: None}),
    )
    case = next(item for item in benchmark.CASES if item.name == family)
    operator = case.factory(**case.kwargs(3))
    inputs = benchmark.inputs_for(operator, case, 3, torch.device("cpu"))
    values = runner._fla_call(case, inputs, "prefill").prepare(inputs)
    assert all(values[key].dtype == torch.float32 for key in ("query", "key", "value"))
    query = inputs["query"].transpose(1, 2)
    expected_query = (
        query.float()
        if family == "rwkv7"
        else (query.float() * case.dim**-0.5).to(query.dtype).float()
    )
    torch.testing.assert_close(values["query"], expected_query)
    if family == "rwkv7":
        beta = inputs["beta"].transpose(1, 2).unsqueeze(-1)
        torch.testing.assert_close(
            values["value"], inputs["value"].transpose(1, 2).float() * beta
        )
        torch.testing.assert_close(
            values["a"],
            -inputs["removal"].transpose(1, 2).float()
            * inputs["decay"].transpose(1, 2).exp(),
        )
    else:
        torch.testing.assert_close(
            values["key"],
            (inputs["key"].transpose(1, 2).float() + 0.1)
            .to(inputs["key"].dtype)
            .float(),
        )


def test_gla_prefill_short_call_pads_neutral_tokens_and_trims(monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark

    initial = torch.randn(1, 8, 64, 64)
    final = torch.randn_like(initial)

    def chunk(query, key, value, gate, **kwargs):
        assert query.shape[1] == 64
        for tensor in (query, key, value, gate):
            assert torch.count_nonzero(tensor[:, 32:]) == 0
        assert kwargs["initial_state"] is initial
        assert kwargs["output_final_state"] is True
        return value, final

    monkeypatch.setitem(sys.modules, "fla.ops.gla", SimpleNamespace(chunk_gla=chunk))
    case = next(item for item in benchmark.CASES if item.name == "gla")
    operator = case.factory(**case.kwargs(32))
    inputs = benchmark.inputs_for(operator, case, 32, torch.device("cpu"))
    adapter = runner._fla_call(case, inputs, "prefill")
    output, state = adapter.invoke(adapter.prepare(inputs), initial, True)
    torch.testing.assert_close(output, inputs["value"].float())
    assert state is final


def test_csv_preserves_baseline_identity_for_unavailable_comparison():
    from testing.benchmark_fla_stateful import _csv_rows

    row = _csv_rows(
        [
            {
                "family": "abc",
                "fla_baseline": "recurrent",
                "fla_function": None,
                "status": "NOT_COMPARABLE",
                "fla_status": "UNAVAILABLE",
            }
        ]
    )[0]
    assert row["fla_baseline"] == "recurrent"
    assert row["fla_function"] is None
    assert row["speedup"] is None


@pytest.mark.parametrize(
    "family", ["preconditioned-kda", "preconditioned-gdn", "lightnet"]
)
@pytest.mark.parametrize("contract", ["output-only", "final-state", "continuation"])
@pytest.mark.parametrize("baseline", ["recurrent", "prefill"])
def test_staged_families_share_public_measurement_contract(
    monkeypatch, family, contract, baseline
):
    """Exercise compiled stages and cache conversion through the main runner."""
    import sys
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark
    from testing.benchmark_fla_stages import oracle

    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    case = replace(
        next(c for c in benchmark.CASES if c.name == family),
        batch=2,
        heads=2,
        dim=3,
        dim_value=4,
    )
    calls = []

    def function(**kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["q"].shape[:3] == (2, kwargs["v"].shape[1], 2)
        assert kwargs["scale"] == 3**-0.5
        native = {
            "query": kwargs["q"].transpose(1, 2),
            "value": kwargs["v"].transpose(1, 2),
        }
        memory = torch.zeros(2, 2, 3, 4)
        if family == "lightnet":
            native["logits"] = kwargs["k"].transpose(1, 2)
            initial = kwargs["initial_state"]
            auxiliary = torch.full((2, 2, 3), -torch.inf)
            if initial is not None:
                memory, z = initial
                assert z.shape == (2, 1, 2, 3)
                auxiliary = z.squeeze(1)
        else:
            native.update(
                {
                    n: kwargs[k].transpose(1, 2)
                    for n, k in (
                        ("key", "k"),
                        ("log_decay", "g"),
                        ("log_atk_decay", "g_atk"),
                        ("beta_atk", "beta_atk"),
                        ("beta", "beta"),
                    )
                }
            )
            shape = native["query"].shape[:-1]
            native["center"] = torch.full(shape, -0.2)
            native["scale"] = torch.full(shape, 3**-0.5)
            auxiliary = kwargs["initial_A_state"]
            memory = kwargs["initial_state"]
            if auxiliary is None:
                assert memory is None
                auxiliary, memory = torch.zeros(2, 2, 3), torch.zeros(2, 2, 3, 4)
            else:
                assert auxiliary.shape == (2, 2, 3) and memory.shape == (2, 2, 3, 4)
            if family == "preconditioned-gdn":
                assert kwargs["use_qk_l2norm_in_kernel"] is True
            elif baseline == "recurrent":
                assert kwargs["use_qk_l2norm_in_kernel"] is False
        calls.append(
            (kwargs["output_final_state"], kwargs["initial_state"] is not None)
        )
        result = oracle(torch, family, native, (auxiliary, memory))
        output = result["output"].transpose(1, 2)
        final = kwargs["output_final_state"]
        auxiliary = result["z" if family == "lightnet" else "atk"]
        if family == "lightnet":
            return output, (result["memory"] if final else None, auxiliary.unsqueeze(1))
        return output, result["memory"] if final else None, auxiliary if final else None

    module, recurrent, prefill = runner.BASELINE_APIS[family]
    monkeypatch.setitem(
        sys.modules,
        f"fla.ops.{module}",
        SimpleNamespace(**{recurrent: function, prefill: function}),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")

    def measure(call, warmup, repetitions, timing):
        assert torch.is_inference_mode_enabled()
        output, states = call()
        assert output["output"].shape == (2, 2, 5, 4)
        assert (states is None) == (contract == "output-only")
        return {"public_cuda_event_ms": 1.0, "cpu_wall_ms": 2.0, "graph_ms": 0.5}

    monkeypatch.setattr(benchmark, "_measure_call", measure)
    result = runner._measure_pair(
        benchmark,
        torch,
        case,
        5,
        torch.device("cpu"),
        1,
        2,
        contract,
        "both",
        workspace_bytes=8 << 30,
        baseline=baseline,
    )
    assert result["meta_correctness"]["status"] == "VALID"
    assert result["meta_lowerer"] == "scan-stage-dag"
    assert len(result["selected_child_lowerers"]) == 2
    assert result["staged_workspace_estimate_bytes"] > 0
    assert result["fla_preprocessing_timed"] is True
    if family == "preconditioned-kda" and baseline == "prefill":
        assert result["status"] == "NOT_COMPARABLE"
        assert result["fla_contract_comparable"] is False
        assert "normalization" in result["fla_note"]
        assert "winner" not in result
        assert runner._csv_rows([result])[0]["speedup"] is None
    else:
        assert result["status"] == result["fla_status"] == "OK"
        assert runner._valid_pair(result)
        assert runner._csv_rows([result])[0]["meta_ms"] == 1.0
    if contract == "continuation":
        assert (True, True) in calls  # Actual prefix caches reached the suffix.
    elif contract == "output-only":
        assert all(not final for final, _ in calls)


@pytest.mark.parametrize(
    "family", ["preconditioned-kda", "preconditioned-gdn", "lightnet"]
)
def test_staged_workloads_preserve_full_batch_head_contract_matrix(family):
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark

    workloads = runner._workloads(
        benchmark, [family], [1, 2], [1, 8], [63, 65], runner.P0_CONTRACTS
    )
    assert len(workloads) == 24
    assert {
        (case.name, case.batch, case.heads, length, contract)
        for case, length, contract in workloads
    } == set(
        runner.benchmark_matrix([family], [1, 2], [1, 8], [63, 65], runner.P0_CONTRACTS)
    )


def test_staged_workspace_rejection_uses_existing_result_status(monkeypatch):
    from dataclasses import replace
    import torch
    from testing import benchmark_fla_stateful as runner
    from testing import benchmark_stateful_plans as benchmark

    monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    case = replace(
        next(c for c in benchmark.CASES if c.name == "lightnet"),
        heads=1,
        dim=3,
        dim_value=4,
    )
    result = runner._measure_pair(
        benchmark,
        torch,
        case,
        2048,
        torch.device("cpu"),
        0,
        1,
        "final-state",
        "e2e",
        workspace_bytes=1024,
        baseline="prefill",
    )
    assert result["status"] == "META_RUNTIME_REJECTED"
    assert result["fla_status"] == "NOT_RUN"
    assert "RUNTIME_RESOURCE" in result["error"]
    row = runner._csv_rows([result])[0]
    assert row["status"] == "META_RUNTIME_REJECTED" and row["speedup"] is None
