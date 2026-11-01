"""Run resumable PR validation. See testing/README-pr-matrix.md for commands.

This controller uses only the standard library. Each child runs in a fresh
process; a successful collection is distinct from a correct or fast result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.benchmark_fla_stateful import (  # noqa: E402
    FLA_FAMILIES,
    _physical_gpu_uuid,
    _valid_pair,
    _visible_physical_indices,
    choose_idle_gpu,
    list_gpus,
)

STAGES = (
    "cpu",
    "gpu-tests",
    "artifacts",
    "fla",
    "generalization",
    "compositions",
    "stages",
    "plans",
)


@dataclass
class Job:
    name: str
    stage: str
    command: list[str]
    artifact: str | None = None
    expected: int = 0
    gpu: bool = False


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def git(*args, cwd=ROOT):
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def source_hash(directory, subdirectories):
    digest = hashlib.sha256()
    # Ignore generated kernel caches, including ignored Python files.
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=directory,
        )
        .decode()
        .split("\0")
    )
    for name in sorted(set(names)):
        path = directory / name
        if (
            path.suffix in (".py", ".json")
            and path.is_file()
            and any(
                name.startswith(subdirectory + "/") for subdirectory in subdirectories
            )
        ):
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def build_jobs(args, directory, smoke):
    jobs = []
    py = sys.executable

    def add(name, stage, command, artifact=None, expected=0, gpu=False):
        if stage in args.stages:
            jobs.append(Job(name, stage, [py, *command], artifact, expected, gpu))

    def pytest_job(name, stage, paths, gpu=False, extra=()):
        artifact = f"{name}/junit.xml"
        add(
            name,
            stage,
            [
                "-m",
                "pytest",
                *paths,
                *extra,
                "-q",
                "-ra",
                f"--junitxml={directory / artifact}",
            ],
            artifact,
            gpu=gpu,
        )

    if smoke:
        pytest_job(
            "cpu-smoke",
            "cpu",
            [
                "tests/unit/test_benchmark_stateful_matrix.py",
                "tests/unit/test_benchmark_fla_stateful.py",
                "tests/unit/test_stateful_generalization.py",
                "tests/unit/test_pr_matrix_runner.py",
            ],
        )
        for script in (
            "benchmark_fla_stateful",
            "benchmark_stateful_generalization",
            "benchmark_stateful_compositions",
            "benchmark_fla_stages",
            "benchmark_stateful_plans",
        ):
            add(f"help-{script}", "cpu", ["-m", f"testing.{script}", "--help"])
        add(
            "generalization-dry-run",
            "cpu",
            [
                "-m",
                "testing.benchmark_stateful_generalization",
                "--dry-run",
                "--levels",
                "0",
                "1",
                "--seeds",
                "1701",
                "--lengths",
                "65",
            ],
        )
        if args.cpu_only:
            return jobs
        pytest_job(
            "gpu-smoke",
            "gpu-tests",
            ["tests/functional/test_stateful_generalization.py"],
            True,
            ("--run-gpu", "-k", "expression-1-1701"),
        )
    else:
        pytest_job(
            "cpu-tests",
            "cpu",
            ["tests"],
            extra=("-m", "not gpu and not benchmark and not slow"),
        )
        # Separate processes also isolate failures and compiler caches per file.
        for path in sorted((ROOT / "tests/functional").glob("test_*.py")):
            pytest_job(
                f"gpu-{path.stem}",
                "gpu-tests",
                [str(path)],
                True,
                ("--run-gpu", "-m", "gpu"),
            )
        pytest_job(
            "gpu-unit-tests",
            "gpu-tests",
            ["tests/unit"],
            True,
            ("--run-gpu", "-m", "gpu"),
        )
        # Hardware fixtures retain explicit SKIP reasons for inapplicable devices.
        for hardware in ("h100", "mi250"):
            pytest_job(
                f"artifacts-{hardware}",
                "artifacts",
                [f"tests/benchmarks/test_{hardware}.py"],
                True,
                ("--run-gpu", "--run-benchmarks"),
            )

    families = ("linear",) if smoke else FLA_FAMILIES
    batches, heads = ((1,), (1,)) if smoke else ((1, 2), (1, 8, 16))
    length_groups = ((65,),) if smoke else ((63, 64, 65, 130), (2048,), (8192,), (1,))
    warmup, reps = (1, 2) if smoke else (10, 30)
    for family in families:
        for batch in batches:
            for head in heads:
                for lengths in length_groups:
                    name = (
                        f"fla-{family}-b{batch}-h{head}-t{'_'.join(map(str, lengths))}"
                    )
                    contracts = (
                        ("output-only", "final-state") if lengths == (1,) else ("all",)
                    )
                    for contract in contracts:
                        task = name + "-" + contract
                        artifact = f"{task}/records.json"
                        expected = len(lengths) * (3 if contract == "all" else 1) * 2
                        add(
                            task,
                            "fla",
                            [
                                "-m",
                                "testing.benchmark_fla_stateful",
                                "--families",
                                family,
                                "--batches",
                                str(batch),
                                "--heads",
                                str(head),
                                "--seqlens",
                                *map(str, lengths),
                                "--contract",
                                contract,
                                "--fla-baseline",
                                "both",
                                "--timing",
                                "both",
                                "--workspace-gib",
                                "1",
                                "--warmup",
                                str(warmup),
                                "--repetitions",
                                str(reps),
                                "--gpu",
                                "{gpu}",
                                "--fla-source",
                                str(args.fla_source),
                                "--output-dir",
                                str(directory / task),
                            ],
                            artifact,
                            expected,
                            True,
                        )

    mutations = (
        ("expression",)
        if smoke
        else ("expression", "propagation", "rank", "write", "readout", "combined")
    )
    shapes = ((1, 4, 64),) if smoke else ((1, 4, 64), (2, 8, 64), (1, 4, 128))
    for family in mutations:
        for batch, head, dim in shapes:
            for length in (65,) if smoke else (65, 512, 2048):
                name = f"generalization-{family}-b{batch}-h{head}-d{dim}-t{length}"
                levels = (0, 1) if smoke else tuple(range(9))
                seeds = (1701,) if smoke else (1701, 2028)
                add(
                    name,
                    "generalization",
                    [
                        "-m",
                        "testing.benchmark_stateful_generalization",
                        "--families",
                        family,
                        "--levels",
                        *map(str, levels),
                        "--seeds",
                        *map(str, seeds),
                        "--lengths",
                        str(length),
                        "--batch",
                        str(batch),
                        "--heads",
                        str(head),
                        "--rows",
                        str(dim),
                        "--columns",
                        str(dim),
                        "--gpu",
                        "0",
                        "--workspace-gib",
                        "1",
                        "--warmup",
                        "1" if smoke else "3",
                        "--repetitions",
                        "2" if smoke else "5",
                        "--batches",
                        "4" if smoke else "10",
                        "--output-dir",
                        str(directory / name),
                    ],
                    f"{name}/records.json",
                    len(levels) * len(seeds),
                    True,
                )

    for batch in (1,) if smoke else (1, 2):
        for length in (65,) if smoke else (65, 512, 2048):
            name = f"compositions-b{batch}-t{length}"
            # The legacy runner does not expose a seed flag. Seed before its main.
            code = "import torch; torch.manual_seed(42); from testing.benchmark_stateful_compositions import main; main()"
            add(
                name,
                "compositions",
                [
                    "-c",
                    code,
                    "--batch",
                    str(batch),
                    "--length",
                    str(length),
                    "--warmup",
                    "1" if smoke else "3",
                    "--repetitions",
                    "2" if smoke else "5",
                    "--batches",
                    "10",
                    "--output",
                    str(directory / name / "records.json"),
                ],
                f"{name}/records.json",
                4,
                True,
            )

    for family in (
        ("lightnet",)
        if smoke
        else ("preconditioned-kda", "preconditioned-gdn", "lightnet")
    ):
        for dim in (64,) if smoke else (64, 128):
            for head in (1,) if smoke else (1, 8):
                for length in (1,) if smoke else (1, 64, 2048, 8192):
                    name = f"stages-{family}-d{dim}-h{head}-t{length}"
                    add(
                        name,
                        "stages",
                        [
                            "-m",
                            "testing.benchmark_fla_stages",
                            "--family",
                            family,
                            "--seqlens",
                            str(length),
                            "--heads",
                            str(head),
                            "--dim",
                            str(dim),
                            "--dtype",
                            "bfloat16",
                            "--workspace",
                            str(1 << 30),
                            "--warmup",
                            "1" if smoke else "3",
                            "--reps",
                            "2" if smoke else "30",
                            "--gpu",
                            "{gpu}",
                            "--fla-path",
                            str(args.fla_source),
                            "--output",
                            str(directory / name / "records.json"),
                        ],
                        f"{name}/records.json",
                        1,
                        True,
                    )

    for family in ("linear",) if smoke else ("mamba2", "pressure"):
        for length in (65,) if smoke else (63, 64, 65, 130, 2048, 8192):
            name = f"plans-{family}-t{length}"
            add(
                name,
                "plans",
                [
                    "-m",
                    "testing.benchmark_stateful_plans",
                    "--families",
                    family,
                    "--seqlen",
                    str(length),
                    "--contract",
                    "all",
                    "--timing",
                    "both",
                    "--warmup",
                    "1" if smoke else "3",
                    "--repetitions",
                    str(reps),
                    "--json",
                    str(directory / name / "records.json"),
                ],
                f"{name}/records.json",
                3,
                True,
            )
    return jobs


def records_for(job, directory):
    value = json.loads((directory / job.artifact).read_text())
    records = value if isinstance(value, list) else value["records"]
    if len(records) != job.expected:
        raise ValueError(f"expected {job.expected} records, got {len(records)}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("record must be an object")
    if job.stage != "compositions" and any(
        "status" not in r and "backends" not in r for r in records
    ):
        raise ValueError("missing result status")
    if len(
        {
            json.dumps(
                {
                    k: r.get(k)
                    for k in (
                        "family",
                        "level",
                        "seed",
                        "batch",
                        "heads",
                        "rows",
                        "columns",
                        "length",
                        "seqlen",
                        "contract",
                        "fla_baseline",
                        "kind",
                    )
                },
                sort_keys=True,
            )
            for r in records
        }
    ) != len(records):
        raise ValueError("duplicate matrix records")
    return records


def validate(job, directory):
    if not job.artifact:
        return
    if job.artifact.endswith(".xml"):
        suites = ET.parse(directory / job.artifact).getroot().findall(".//testsuite")
        if not suites or sum(int(s.get("tests", 0)) for s in suites) == 0:
            raise ValueError("pytest selected zero tests")
    else:
        records_for(job, directory)


def child_environment():
    env = dict(
        os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    # Invoking .venv/bin/python does not activate its console scripts. Native
    # extension builders find Ninja via PATH, including in nested subprocesses.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(ROOT / "attention_engine")
    return env


def external_gpu_activity(gpu):
    # Utilization averages may still include this child's completed kernels.
    # After it exits, external processes and resident memory are the evidence.
    return (
        gpu.query_error is not None
        or bool(gpu.compute_pids)
        or (gpu.memory_used_mib or 0) > 1024
    )


def verify_gpu_mapping(index, env):
    expected = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    code = "import torch; print(torch.cuda.get_device_properties(0).uuid)"
    actual = subprocess.check_output(
        [sys.executable, "-c", code], env=env, text=True, timeout=30
    ).strip()
    if expected.removeprefix("GPU-").lower() != actual.removeprefix("GPU-").lower():
        raise RuntimeError(
            f"physical GPU {index} UUID {expected} does not match CUDA device 0 UUID {actual}; refusing to execute on a different GPU"
        )
    return expected


def execute(command, env, log, timeout):
    with log.open("w") as output:
        output.write(shlex.join(command) + "\n")
        output.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def summarize(directory, jobs, state):
    lines = [
        "# PR validation matrix",
        "",
        "Collection completion does not imply correctness or performance stability.",
        "Smoke results are exploratory and must not be used as performance claims.",
        "",
        "| Job | Collection | Exit | Result counts |",
        "| --- | --- | --- | --- |",
    ]
    all_records = []
    tests = []
    for job in jobs:
        result = state.get(job.name, {})
        counts = Counter()
        if job.artifact and (directory / job.artifact).exists():
            try:
                if job.artifact.endswith(".xml"):
                    for case in (
                        ET.parse(directory / job.artifact).getroot().iter("testcase")
                    ):
                        child = next(
                            (
                                c
                                for c in case
                                if c.tag in ("failure", "error", "skipped")
                            ),
                            None,
                        )
                        status = "PASS" if child is None else child.tag.upper()
                        if child is not None and child.get("type") == "pytest.xfail":
                            status = "XFAIL"
                        counts[status] += 1
                        tests.append(
                            {
                                "job": job.name,
                                "name": case.get("name"),
                                "classname": case.get("classname"),
                                "status": status,
                                "details": ET.tostring(child, encoding="unicode")
                                if child is not None
                                else None,
                            }
                        )
                else:
                    for record in records_for(job, directory):
                        counts[
                            record.get(
                                "status",
                                "BACKENDS"
                                if "backends" in record
                                else "MEASURED_UNVALIDATED",
                            )
                        ] += 1
                        for backend, entry in record.get("backends", {}).items():
                            counts[f"{backend}:{entry.get('status', 'UNKNOWN')}"] += 1
                        # Legacy composition timing has no independent in-run oracle.
                        if job.stage == "compositions":
                            record["correctness_status"] = "NOT_CHECKED_BY_RUNNER"
                        all_records.append(
                            {
                                "job": job.name,
                                "stage": job.stage,
                                "collection_status": result.get("status", "PENDING"),
                                "record": record,
                            }
                        )
            except (ValueError, KeyError, ET.ParseError) as error:
                counts[f"ARTIFACT_ERROR: {error}"] += 1
        lines.append(
            f"| {job.name} | {result.get('status', 'PENDING')} | {result.get('returncode', '')} | {dict(counts)} |"
        )
    artifact_records = []
    for job in jobs:
        if job.stage != "artifacts":
            continue
        measurement_path = directory / job.name / "h100-results" / "measurements.json"
        if measurement_path.exists():
            for record in json.loads(measurement_path.read_text()):
                artifact_records.append({"job": job.name, **record})
    save(directory / "hardware_measurements.json", artifact_records)
    if artifact_records:
        counts = Counter(record["status"] for record in artifact_records)
        lines += [
            "",
            f"Hardware artifact measurements: {dict(counts)}.",
            "Raw Figure 11 measurements and limitation reasons are retained in hardware_measurements.json; measured latencies have no independent in-run correctness oracle.",
            "",
            "| Case | Method | Direction | Status | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in artifact_records:
            if record["status"] in {"RESOURCE_LIMIT", "UNSUPPORTED"}:
                reason = json.dumps(record["reason"], sort_keys=True).replace(
                    "|", "\\|"
                )
                lines.append(
                    f"| {record['case_id']} | {record['method']} | {record['direction']} | {record['status']} | {reason} |"
                )
    valid = [
        item["record"]
        for item in all_records
        if item["stage"] == "fla"
        and item["collection_status"] == "COLLECTED"
        and _valid_pair(item["record"])
    ]
    lines += [
        "",
        f"Accuracy-qualified FLA comparisons: {len(valid)}.",
        "Ratios use FLA / Meta; >1 means Meta is faster. No PR-base regression is inferred.",
        "",
        "| Family | Contract | Schedule | Valid pairs | CUDA speedup range |",
        "| --- | --- | --- | --- | --- |",
    ]
    groups = {}
    for record in valid:
        groups.setdefault(
            (record["family"], record["contract"], record["fla_baseline"]), []
        ).append(record)
    for key, values in sorted(groups.items()):
        ratios = [
            r["speedup_public_cuda_event_ms"]
            for r in values
            if isinstance(r.get("speedup_public_cuda_event_ms"), (int, float))
        ]
        span = f"{min(ratios):.3f}–{max(ratios):.3f}" if ratios else "unavailable"
        lines.append(f"| {' | '.join(key)} | {len(values)} | {span} |")
    lines += [
        "",
        "| Generalization job | Family | B/H/K/V/T | Stable levels | First boundary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for job in jobs:
        if job.stage != "generalization" or not (directory / job.artifact).exists():
            continue
        payload = json.loads((directory / job.artifact).read_text())
        for group in payload.get("envelopes", []):
            if state.get(job.name, {}).get("status") != "COLLECTED":
                continue
            shape = "/".join(
                str(group[k]) for k in ("batch", "heads", "rows", "columns", "length")
            )
            lines.append(
                f"| {job.name} | {group['family']} | {shape} | {group['stable_prefix_levels']} | {group['first_unstable_or_failed_level']} |"
            )
    lines += [
        "",
        "Generalization envelopes are in each generalization job's report.md; all requested levels and seeds must be present.",
        "Stage backward and composition timings have separate measurement contracts; consult raw records.",
    ]
    fields = (
        "job",
        "stage",
        "collection_status",
        "family",
        "batch",
        "heads",
        "dim",
        "rows",
        "columns",
        "seqlen",
        "length",
        "contract",
        "fla_baseline",
        "level",
        "seed",
        "status",
        "meta_ms",
        "fla_ms",
        "speedup",
        "stable",
        "error",
    )
    with (directory / "matrix.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in all_records:
            record = item["record"]
            row = {field: record.get(field) for field in fields}
            row.update(
                {key: item[key] for key in ("job", "stage", "collection_status")}
            )
            row.update(
                meta_ms=record.get("meta", {}).get("public_cuda_event_ms"),
                fla_ms=record.get("fla", {}).get("public_cuda_event_ms"),
            )
            if item["stage"] == "generalization":
                row["meta_ms"] = (
                    record.get("variant", {}).get("cuda_ms", {}).get("median")
                )
            if item["collection_status"] == "COLLECTED" and _valid_pair(record):
                row["speedup"] = record.get("speedup_public_cuda_event_ms")
            writer.writerow(row)
    save(directory / "matrix.json", all_records)
    save(directory / "tests.json", tests)
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "smoke", "full", "report"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="full",
        help="profile for plan mode",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="smoke profile only; no GPU validation"
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument(
        "--gpu", type=int, help="physical index; otherwise select an idle GPU"
    )
    parser.add_argument(
        "--fla-source", type=Path, default=ROOT.parent / "flash-linear-attention"
    )
    parser.add_argument(
        "--timeout", type=int, default=14400, help="seconds per child process"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry failed collections, retaining attempt logs",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.fla_source = args.fla_source.resolve()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    smoke = args.mode == "smoke" or (args.mode == "plan" and args.profile == "smoke")
    if args.cpu_only and not smoke:
        parser.error("--cpu-only requires smoke or plan --profile smoke")
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    # Prevent two controllers from sharing a run or racing its checkpoint writes.
    with (directory / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another controller owns this output directory")
        if args.mode == "report":
            manifest = json.loads((directory / "manifest.json").read_text())
            jobs = [Job(**job) for job in manifest["jobs"]]
            summarize(
                directory, jobs, json.loads((directory / "state.json").read_text())
            )
            return
        jobs = build_jobs(args, directory, smoke)
        if not jobs:
            parser.error("no jobs selected")
        config = {
            "profile": "smoke" if smoke else "full",
            "cpu_only": args.cpu_only,
            "gpu_request": args.gpu,
            "python": sys.executable,
            "python_version": sys.version,
            "timeout": args.timeout,
            "git_commit": git("rev-parse", "HEAD"),
            "git_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
            ).hexdigest(),
            "source_sha256": source_hash(
                ROOT,
                ("attention_engine", "examples", "testing", "tests", "tuned_config"),
            ),
            "versions": {},
            "project_config_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("pyproject.toml", "uv.lock")
            },
            "fla_source": str(args.fla_source),
            "fla_commit": git("rev-parse", "HEAD", cwd=args.fla_source),
            "fla_source_sha256": source_hash(args.fla_source, ("fla",)),
            "jobs": [asdict(job) for job in jobs],
        }
        for package in (
            "torch",
            "tilelang",
            "triton",
            "pytest",
            "flash-linear-attention",
            "ninja",
            "nvidia-cuda-cccl-cu12",
            "nvidia-cuda-nvcc-cu12",
        ):
            try:
                config["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                config["versions"][package] = None
        if args.mode == "plan":
            save(directory / "plan.json", config)
            print(
                json.dumps(
                    {
                        "jobs": len(jobs),
                        "stages": dict(Counter(job.stage for job in jobs)),
                        "expected_benchmark_records": dict(
                            Counter(
                                {
                                    stage: sum(
                                        j.expected for j in jobs if j.stage == stage
                                    )
                                    for stage in STAGES
                                }
                            )
                        ),
                        "plan": str(directory / "plan.json"),
                    },
                    indent=2,
                )
            )
            return
        previous = directory / "manifest.json"
        if previous.exists():
            manifest = json.loads(previous.read_text())
            if manifest["config"] != config:
                parser.error(
                    "run configuration or sources changed; use a new output directory"
                )
        else:
            manifest = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "jobs": config["jobs"],
                "git_status": git("status", "--short"),
                "gpu_runs": [],
            }
        save(previous, manifest)
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        save(state_path, state)
        gpu = (
            manifest["gpu_runs"][0]["before"]["index"] if manifest["gpu_runs"] else None
        )
        mapping_verified = False
        try:
            for index, job in enumerate(jobs, 1):
                old = state.get(job.name, {})
                if old.get("status") == "COLLECTED":
                    validate(job, directory)
                    continue
                if old.get("status") == "FAILED" and not args.retry_failed:
                    if job.stage in ("cpu", "gpu-tests", "artifacts"):
                        raise RuntimeError(
                            f"test gate failed: {job.name}; use --retry-failed after resolving it"
                        )
                    continue
                env = child_environment()
                if job.gpu:
                    if os.environ.get("STATEFUL_OPERATOR_TEST_CPU"):
                        raise RuntimeError(
                            "unset STATEFUL_OPERATOR_TEST_CPU before GPU execution"
                        )
                    snapshot = list_gpus()
                    # Let utilization from our previous child leave the sampled
                    # window; never wait through an observed external process.
                    if gpu is not None:
                        for _ in range(5):
                            current = next(g for g in snapshot if g.index == gpu)
                            if current.idle or external_gpu_activity(current):
                                break
                            time.sleep(1)
                            snapshot = list_gpus()
                    selected = choose_idle_gpu(
                        snapshot,
                        requested=gpu if gpu is not None else args.gpu,
                        allowed_indices=_visible_physical_indices(),
                    )
                    gpu = selected.index
                    manifest["gpu_runs"].append(
                        {
                            "job": job.name,
                            "before": asdict(selected),
                            "time": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    save(previous, manifest)
                    env["CUDA_VISIBLE_DEVICES"] = _physical_gpu_uuid(gpu)
                    if not mapping_verified:
                        manifest["selected_gpu_uuid"] = verify_gpu_mapping(gpu, env)
                        save(previous, manifest)
                        mapping_verified = True
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                command = [
                    str(gpu) if token == "{gpu}" else token for token in job.command
                ]
                task_dir = directory / job.name
                task_dir.mkdir(parents=True, exist_ok=True)
                if job.stage == "artifacts":
                    env["PR_MATRIX_ARTIFACT_DIR"] = str(task_dir)
                attempt = old.get("attempt", 0) + 1
                result = {"status": "RUNNING", "attempt": attempt, "command": command}
                state[job.name] = result
                save(state_path, state)
                print(f"[{index}/{len(jobs)}] {job.name}", flush=True)
                started = time.monotonic()
                if job.artifact:
                    (directory / job.artifact).unlink(missing_ok=True)
                try:
                    result["returncode"] = execute(
                        command, env, task_dir / f"attempt-{attempt}.log", args.timeout
                    )
                    validate(job, directory)
                    result["status"] = (
                        "COLLECTED" if result["returncode"] == 0 else "FAILED"
                    )
                except (
                    subprocess.TimeoutExpired,
                    ValueError,
                    OSError,
                    KeyError,
                    ET.ParseError,
                ) as error:
                    result.update(status="FAILED", error=str(error))
                result["elapsed_seconds"] = time.monotonic() - started
                if job.gpu:
                    after = next(g for g in list_gpus() if g.index == gpu)
                    result["gpu_after"] = asdict(after)
                    if external_gpu_activity(after):
                        result.update(
                            status="FAILED",
                            error="GPU occupied after child; timing not qualified",
                        )
                save(state_path, state)
                if result["status"] != "COLLECTED" and job.stage in (
                    "cpu",
                    "gpu-tests",
                    "artifacts",
                ):
                    raise RuntimeError(
                        f"test gate failed: {job.name}; inspect its attempt log before benchmarking"
                    )
        except (RuntimeError, KeyboardInterrupt) as error:
            print(f"Stopped: {error}. Completed jobs are retained.", file=sys.stderr)
            summarize(directory, jobs, state)
            return 2
        summarize(directory, jobs, state)
        incomplete = sum(
            state.get(j.name, {}).get("status") != "COLLECTED" for j in jobs
        )
        print(
            f"Report: {directory / 'report.md'}; incomplete/failed collections: {incomplete}"
        )
        return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
