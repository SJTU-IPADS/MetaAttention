"""Controller contracts that do not need a GPU or benchmark execution."""

import argparse
from argparse import Namespace
from dataclasses import asdict
import importlib
import json
import subprocess
import sys

import pytest

from testing.run_pr_matrix import (
    Job,
    ROOT,
    STAGES,
    build_jobs,
    child_environment,
    execute,
    external_gpu_activity,
    records_for,
    source_hash,
    summarize,
    validate,
    verify_gpu_mapping,
)

pytestmark = pytest.mark.unit


def arguments(**overrides):
    return Namespace(
        **dict(
            {
                "stages": list(STAGES),
                "cpu_only": False,
                "fla_source": ROOT.parent / "flash-linear-attention",
            },
            **overrides,
        )
    )


def test_full_plan_covers_matrix_and_gpu_unit_tests(tmp_path):
    jobs = build_jobs(arguments(), tmp_path, False)
    assert len({job.name for job in jobs}) == len(jobs)
    assert sum(j.expected for j in jobs if j.stage == "fla") == 6000
    assert sum(j.expected for j in jobs if j.stage == "generalization") == 972
    assert any(j.name == "gpu-unit-tests" and j.gpu for j in jobs)
    functional = {p.stem for p in (ROOT / "tests/functional").glob("test_*.py")}
    assert {
        j.name.removeprefix("gpu-") for j in jobs if j.name.startswith("gpu-test_")
    } == functional
    for job in jobs:
        if (
            job.stage == "fla"
            and "1" == job.command[job.command.index("--seqlens") + 1]
        ):
            assert job.command[job.command.index("--contract") + 1] != "all"


def test_every_benchmark_command_matches_its_real_cli(tmp_path, monkeypatch):
    class Parsed(Exception):
        pass

    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(parser, *args, **kwargs):
        original(parser, *args, **kwargs)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    for job in build_jobs(arguments(), tmp_path, False):
        if not job.expected:
            continue
        module = (
            "testing.benchmark_stateful_compositions"
            if job.stage == "compositions"
            else job.command[2]
        )
        main = importlib.import_module(module).main
        argv = [module, *("0" if v == "{gpu}" else v for v in job.command[3:])]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(Parsed):
            main()


def test_source_fingerprint_ignores_generated_caches(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("cache/\n")
    directory = tmp_path / "testing"
    directory.mkdir()
    (directory / "runner.py").write_text("original\n")
    original = source_hash(tmp_path, ("testing",))
    cache = directory / "cache"
    cache.mkdir()
    (cache / "generated.py").write_text("generated\n")
    assert source_hash(tmp_path, ("testing",)) == original
    (directory / "runner.py").write_text("changed\n")
    assert source_hash(tmp_path, ("testing",)) != original


def test_source_fingerprint_includes_tuning_configuration(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    directory = tmp_path / "tuned_config"
    directory.mkdir()
    config = directory / "relu.json"
    config.write_text('{"block_M": 256}\n')
    original = source_hash(tmp_path, ("tuned_config",))
    config.write_text('{"block_M": 128}\n')
    assert source_hash(tmp_path, ("tuned_config",)) != original


def test_cpu_smoke_cannot_schedule_gpu_work(tmp_path):
    jobs = build_jobs(arguments(cpu_only=True), tmp_path, True)
    assert len(jobs) == 7
    assert not any(j.gpu for j in jobs)


def test_record_validation_rejects_partial_and_duplicate_matrix(tmp_path):
    job = Job("fla", "fla", [], "records.json", 2, True)
    path = tmp_path / job.artifact
    path.write_text(json.dumps([{"status": "OK", "family": "linear"}]))
    with pytest.raises(ValueError, match="expected 2"):
        records_for(job, tmp_path)
    path.write_text(json.dumps([{"status": "OK", "family": "linear"}] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        records_for(job, tmp_path)
    path.write_text(
        json.dumps(
            [
                {"status": "OK", "family": "linear"},
                {"status": "NOT_COMPARABLE", "family": "gla"},
            ]
        )
    )
    assert len(records_for(job, tmp_path)) == 2


def test_pytest_zero_selection_is_a_collection_failure(tmp_path):
    job = Job("tests", "cpu", [], "junit.xml")
    (tmp_path / job.artifact).write_text(
        '<testsuites><testsuite tests="0" /></testsuites>'
    )
    with pytest.raises(ValueError, match="zero tests"):
        validate(job, tmp_path)


def test_report_excludes_failed_or_invalid_speedups_and_keeps_xfail(tmp_path):
    record = {
        "family": "linear",
        "contract": "output-only",
        "fla_baseline": "prefill",
        "status": "OK",
        "fla_status": "OK",
        "fla_contract_comparable": True,
        "meta_correctness": {"status": "VALID"},
        "fla_correctness": {"status": "VALID"},
        "pair_correctness": {"status": "VALID"},
        "speedup_public_cuda_event_ms": 2.0,
    }
    jobs = [
        Job("invalid", "fla", [], "invalid.json", 1),
        Job("failed", "fla", [], "failed.json", 1),
        Job("test", "cpu", [], "junit.xml"),
    ]
    (tmp_path / "invalid.json").write_text(
        json.dumps([{**record, "pair_correctness": {"status": "INVALID"}}])
    )
    (tmp_path / "failed.json").write_text(json.dumps([record]))
    (tmp_path / "junit.xml").write_text(
        '<testsuites><testsuite tests="1"><testcase name="known"><skipped type="pytest.xfail" message="known boundary" /></testcase></testsuite></testsuites>'
    )
    summarize(
        tmp_path,
        jobs,
        {"invalid": {"status": "COLLECTED"}, "failed": {"status": "FAILED"}},
    )
    assert (
        "Accuracy-qualified FLA comparisons: 0" in (tmp_path / "report.md").read_text()
    )
    assert json.loads((tmp_path / "tests.json").read_text())[0]["status"] == "XFAIL"
    assert ",2.0," not in (tmp_path / "matrix.csv").read_text()


def test_child_finds_virtualenv_build_tools_without_activation(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = child_environment()
    assert (
        execute(
            [
                sys.executable,
                "-c",
                "import shutil, sys; from pathlib import Path; assert shutil.which('ninja') == str(Path(sys.executable).parent / 'ninja')",
            ],
            env,
            tmp_path / "build-tools.log",
            10,
        )
        == 0
    )
    assert "PATH" in env
    assert (
        execute(
            [sys.executable, "-c", "import shutil; assert shutil.which('ninja')"],
            env,
            tmp_path / "ninja.log",
            10,
        )
        == 0
    )


def test_post_child_utilization_does_not_imply_external_contention():
    from testing.benchmark_fla_stateful import GPUInfo

    gpu = GPUInfo(0, "H20", 4, 97871, 2)
    assert not external_gpu_activity(gpu)
    assert external_gpu_activity(GPUInfo(0, "H20", 4, 97871, 2, (123,)))
    assert external_gpu_activity(GPUInfo(0, "H20", 2048, 97871, 0))
    assert external_gpu_activity(
        GPUInfo(0, "H20", 4, 97871, 0, query_error="query failed")
    )


def test_gpu_mapping_rejects_numeric_index_drift(monkeypatch):
    replies = iter(("GPU-physical-three\n", "physical-four\n"))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: next(replies))
    with pytest.raises(RuntimeError, match="does not match"):
        verify_gpu_mapping(3, {})
    replies = iter(("GPU-physical-zero\n", "physical-zero\n"))
    assert verify_gpu_mapping(0, {}) == "GPU-physical-zero"


def test_physical_gpu_uuid_mask_survives_nested_benchmark_selection(monkeypatch):
    from testing.benchmark_fla_stateful import (
        GPUInfo,
        _physical_gpu_uuid,
        _visible_physical_indices,
        choose_idle_gpu,
    )

    def query(command, **kwargs):
        assert "--id=3" in command
        return subprocess.CompletedProcess(command, 0, "GPU-physical-three\n", "")

    monkeypatch.setattr(subprocess, "run", query)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: "0, GPU-physical-zero\n3, GPU-physical-three\n",
    )
    mask = _physical_gpu_uuid(3)
    assert mask == "GPU-physical-three"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    selected = choose_idle_gpu(
        (GPUInfo(0, "H20", 4, 97871, 0), GPUInfo(3, "H20", 4, 97871, 0)),
        requested=3,
        allowed_indices=_visible_physical_indices(),
    )
    assert selected.index == 3
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-unknown")
    with pytest.raises(RuntimeError, match="unsupported"):
        _visible_physical_indices()


def test_child_timeout_leaves_log_and_reaps_process(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        execute(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('started'); time.sleep(30)",
            ],
            None,
            tmp_path / "timeout.log",
            1,
        )
    assert "started" in (tmp_path / "timeout.log").read_text()


def test_report_cli_reads_checkpoint_without_running_jobs(tmp_path):
    job = Job(
        "would-fail",
        "cpu",
        [sys.executable, "-c", "raise RuntimeError('must not execute')"],
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": [asdict(job)]}))
    (tmp_path / "state.json").write_text("{}")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "testing.run_pr_matrix",
            "report",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PENDING" in (tmp_path / "report.md").read_text()


def test_report_retains_hardware_resource_limits(tmp_path):
    artifact_dir = tmp_path / "artifacts-h100"
    results = artifact_dir / "h100-results"
    results.mkdir(parents=True)
    (artifact_dir / "junit.xml").write_text(
        '<testsuites><testsuite tests="1"><testcase name="artifacts" /></testsuite></testsuites>'
    )
    record = {
        "case_id": "rfa-01",
        "method": "MetaAttention",
        "direction": "backward",
        "status": "RESOURCE_LIMIT",
        "reason": {"workspace_bytes": 1 << 30},
        "latency_ms": None,
    }
    (results / "measurements.json").write_text(json.dumps([record]))
    summarize(
        tmp_path,
        [Job("artifacts-h100", "artifacts", [], "artifacts-h100/junit.xml")],
        {"artifacts-h100": {"status": "COLLECTED"}},
    )
    measurements = json.loads((tmp_path / "hardware_measurements.json").read_text())
    assert measurements == [{"job": "artifacts-h100", **record}]
    report = (tmp_path / "report.md").read_text()
    assert "RESOURCE_LIMIT" in report and "rfa-01" in report
    assert "Accuracy-qualified FLA comparisons: 0" in report
