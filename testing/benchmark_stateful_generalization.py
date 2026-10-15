"""Measure the expression-complexity envelope of unseen Stateful Operators.

Run from the repository root, for example::

    python testing/benchmark_stateful_generalization.py --gpu 3 \
        --levels 0 1 2 4 8 --lengths 65 512 2048 \
        --output-dir benchmark_results/generalization

No named attention factory or optional FLA dependency participates in this suite.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attn_engine import StateTuple, StatefulCompilationError, Target  # noqa: E402
from testing.stateful_generalization_profiles import (  # noqa: E402
    FAMILIES,
    Mutation,
    check_result,
    complexity,
    make_inputs,
    make_operator,
    oracle,
)


def distribution(samples):
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    median = statistics.median(samples)
    return {
        "median": median,
        "relative_iqr": (quartiles[2] - quartiles[0]) / median if median else 0.0,
        "samples": samples,
    }


def assess(control, variant, *, max_slowdown, max_jitter):
    ratios = [v / c for c, v in zip(control, variant, strict=True)]
    result = distribution(ratios)
    return {
        "variant_over_control": result,
        "within_latency_budget": result["median"] <= max_slowdown,
        "repeatable": result["relative_iqr"] <= max_jitter,
        "stable": result["median"] <= max_slowdown
        and result["relative_iqr"] <= max_jitter,
    }


def _call(compiled, values, initial):
    return compiled(
        **values,
        initial_state=StateTuple(("memory",), (initial,)),
        return_final_state=True,
    )


def _measure(call, repetitions):
    cuda, wall = [], []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(repetitions):
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        result = call()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - wall_start) * 1000.0)
        cuda.append(start.elapsed_time(end))
        del result
    return {
        "cuda_ms": statistics.median(cuda),
        "wall_ms": statistics.median(wall),
        "cuda_samples_ms": cuda,
        "wall_samples_ms": wall,
    }


def _peak(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del result
    return peak


def _prepare(mutation, args, length):
    operator, sizes = make_operator(
        mutation, heads=args.heads, rows=args.rows, columns=args.columns
    )
    compiled = operator.compile(
        role_sizes=sizes,
        mode="dense",
        optimized=True,
        target=Target(
            product_family=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            workspace_bytes=int(args.workspace_gib * (1 << 30)),
        ),
    )
    executable = operator._plans[compiled.specialization]
    plan = executable.plan
    values, initial = make_inputs(
        operator,
        sizes,
        batch=args.batch,
        length=length,
        seed=mutation.seed,
        device=torch.device("cuda"),
    )

    def call():
        return _call(compiled, values, initial)

    metadata = {
        "complexity": complexity(operator),
        "lowerer": plan.candidate.lowerer,
        "runtime_family": executable.runtime_family,
        "representation": plan.mathematical_plan.representation.value,
        "rewrites": list(plan.mathematical_plan.rewrites),
        "workspace_bytes": plan.workspace.required_bytes(
            args.batch * length, sequence_count=args.batch, sequence_length=length
        ),
    }
    result = call()  # First-call code generation is excluded from timing.
    expected, expected_state = oracle(operator, sizes, values, initial)
    correctness = {
        "oracle": "independent FP32 CPU token recurrence",
        "output_tolerance": {"rtol": 0.03, "atol": 0.001},
        "state_tolerance": {"rtol": 0.02, "atol": 0.001},
    }
    try:
        correctness.update(
            status="PASS", max_abs_errors=check_result(result, expected, expected_state)
        )
    except AssertionError as error:
        correctness.update(status="FAIL", error=str(error))
    metadata["correctness"] = correctness
    return call, metadata


def benchmark(mutation, args, length):
    record = {
        **asdict(mutation),
        "batch": args.batch,
        "heads": args.heads,
        "rows": args.rows,
        "columns": args.columns,
        "length": length,
    }
    phase = "control"
    try:
        control, control_metadata = _prepare(
            Mutation(mutation.family, 0, mutation.seed), args, length
        )
        record["control"] = control_metadata
        if control_metadata["correctness"]["status"] != "PASS":
            return {
                **record,
                "status": "INCORRECT",
                "stable": False,
                "phase": phase,
                "error": control_metadata["correctness"]["error"],
            }
        phase = "variant"
        variant, variant_metadata = _prepare(mutation, args, length)
        record["variant"] = variant_metadata
        if variant_metadata["correctness"]["status"] != "PASS":
            return {
                **record,
                "status": "INCORRECT",
                "stable": False,
                "phase": phase,
                "error": variant_metadata["correctness"]["error"],
            }
        measurements = {"control": [], "variant": []}
        phase = "measurement"
        with torch.no_grad():
            for call in (control, variant):
                for _ in range(args.warmup):
                    call()
            for index in range(args.batches):
                order = (
                    ("control", "variant") if index % 2 == 0 else ("variant", "control")
                )
                for label in order:
                    measurements[label].append(
                        _measure(
                            control if label == "control" else variant, args.repetitions
                        )
                    )
            for label, call in (("control", control), ("variant", variant)):
                record[label]["peak_incremental_allocated_bytes"] = _peak(call)
        record["measurements"] = measurements
        assessments = {}
        for metric in ("cuda_ms", "wall_ms"):
            samples = {
                label: [batch[metric] for batch in batches]
                for label, batches in measurements.items()
            }
            assessments[metric] = assess(
                samples["control"],
                samples["variant"],
                max_slowdown=args.max_slowdown,
                max_jitter=args.max_jitter,
            )
            for label in samples:
                record[label][metric] = distribution(samples[label])
        record["assessment"] = assessments
        record["stable"] = assessments[args.metric]["stable"]
        record["status"] = "OK"
    except (StatefulCompilationError, RuntimeError, AssertionError) as error:
        record.update(
            status="REJECTED"
            if isinstance(error, StatefulCompilationError)
            else "INCORRECT"
            if isinstance(error, AssertionError)
            else "ERROR",
            stable=False,
            phase=phase,
            error=str(error),
        )
        if isinstance(error, StatefulCompilationError):
            record["rejection"] = error.as_dict()
    return record


def summarize(records):
    """Conservative contiguous envelope: all seeds must pass at each level.

    A later recovery does not erase an earlier cliff. Failed/missing seed points
    cannot count as stable. Each group is one family and one workload shape.
    """
    groups = {}
    for record in records:
        key = tuple(
            record[k] for k in ("family", "batch", "heads", "rows", "columns", "length")
        )
        groups.setdefault(key, []).append(record)
    summaries = []
    for key, points in groups.items():
        seeds = {point["seed"] for point in points}
        levels = sorted({point["level"] for point in points})
        stable_levels, boundary = [], None
        for level in levels:
            at_level = [p for p in points if p["level"] == level]
            passed = {
                p["seed"] for p in at_level if p["status"] == "OK" and p["stable"]
            }
            if passed != seeds:
                boundary = level
                break
            stable_levels.append(level)
        summaries.append(
            {
                **dict(
                    zip(("family", "batch", "heads", "rows", "columns", "length"), key)
                ),
                "seeds": sorted(seeds),
                "tested_levels": levels,
                "stable_prefix_levels": stable_levels,
                "largest_stable_tested_level": stable_levels[-1]
                if stable_levels
                else None,
                "first_unstable_or_failed_level": boundary,
                "boundary_observed": boundary is not None,
            }
        )
    return summaries


def _write(payload, directory):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "records.json"
    temporary = directory / "records.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)
    columns = (
        "family",
        "level",
        "seed",
        "batch",
        "heads",
        "rows",
        "columns",
        "length",
        "status",
        "stable",
        "lowerer",
        "expression_nodes",
        "control_cuda_ms",
        "variant_cuda_ms",
        "cuda_ratio",
        "wall_ratio",
        "ratio_relative_iqr",
        "error",
    )
    with (directory / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in payload["records"]:
            row = {column: record.get(column) for column in columns}
            if record["status"] == "OK":
                variant = record["variant"]
                row.update(
                    lowerer=variant["lowerer"],
                    expression_nodes=variant["complexity"]["expression_nodes"],
                    control_cuda_ms=record["control"]["cuda_ms"]["median"],
                    variant_cuda_ms=variant["cuda_ms"]["median"],
                    cuda_ratio=record["assessment"]["cuda_ms"]["variant_over_control"][
                        "median"
                    ],
                    wall_ratio=record["assessment"]["wall_ms"]["variant_over_control"][
                        "median"
                    ],
                    ratio_relative_iqr=record["assessment"][payload["metric"]][
                        "variant_over_control"
                    ]["relative_iqr"],
                )
            writer.writerow(row)
    lines = [
        "# Stateful expression generalization envelope",
        "",
        f"Metric: `{payload['metric']}`. Stable requires variant/control <= "
        f"{payload['max_slowdown']} and paired ratio IQR/median <= {payload['max_jitter']}.",
        "All seeds must pass. A failure closes the contiguous tested envelope.",
        "A missing boundary only means the largest tested level passed.",
        "",
        "| Family | B/H/K/V/T | Stable tested levels | First unstable/failed |",
        "| --- | --- | --- | --- |",
    ]
    for group in payload["envelopes"]:
        shape = "/".join(
            str(group[k]) for k in ("batch", "heads", "rows", "columns", "length")
        )
        lines.append(
            f"| {group['family']} | {shape} | {group['stable_prefix_levels']} | "
            f"{group['first_unstable_or_failed_level']} |"
        )
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2028])
    parser.add_argument("--lengths", nargs="+", type=int, default=[65, 512, 2048])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--max-slowdown", type=float, default=1.25)
    parser.add_argument("--max-jitter", type=float, default=0.10)
    parser.add_argument("--metric", choices=("cuda_ms", "wall_ms"), default="cuda_ms")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the CPU-only profile matrix"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/generalization")
    )
    args = parser.parse_args()
    if (
        min(
            args.batch,
            args.heads,
            args.rows,
            args.columns,
            *args.lengths,
            args.repetitions,
        )
        < 1
        or min(args.levels) < 0
        or args.warmup < 0
        or args.batches < 4
        or not 0 < args.workspace_gib < float("inf")
        or not 1 <= args.max_slowdown < float("inf")
        or not 0 <= args.max_jitter < float("inf")
    ):
        parser.error(
            "positive shapes/budget/repetitions, nonnegative levels/warmup/jitter, "
            "slowdown >=1, and at least four batches are required"
        )
    args.levels = sorted(set([0, *args.levels]))
    matrix = [
        (Mutation(family, level, seed), length)
        for family in dict.fromkeys(args.families)
        for length in dict.fromkeys(args.lengths)
        for level in args.levels
        for seed in dict.fromkeys(args.seeds)
    ]
    if args.dry_run:
        for mutation, length in matrix:
            operator, _ = make_operator(
                mutation, heads=args.heads, rows=args.rows, columns=args.columns
            )
            print(
                json.dumps(
                    {**asdict(mutation), "length": length, **complexity(operator)}
                )
            )
        return
    if not torch.cuda.is_available():
        parser.error(
            "CUDA is required for measurements; use --dry-run to inspect on CPU"
        )
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(1)
    payload = {
        "schema_version": "stateful-generalization/1",
        "device": torch.cuda.get_device_name(),
        "gpu": args.gpu,
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__)
                .with_name("stateful_generalization_profiles.py")
                .resolve(),
                Path(__file__).with_name("benchmark_stateful_plans.py").resolve(),
            )
        },
        "gpu_snapshot": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tilelang": importlib.metadata.version("tilelang"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "contract": "forward inference, nonzero initial State, all outputs and final State",
        "measurement": "alternating paired batches; median of per-batch variant/control ratios; "
        "compilation, first call and oracle excluded; public layout preparation included",
        "metric": args.metric,
        "max_slowdown": args.max_slowdown,
        "max_jitter": args.max_jitter,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "records": [],
        "envelopes": [],
    }
    for mutation, length in matrix:
        with torch.no_grad():
            record = benchmark(mutation, args, length)
        payload["records"].append(record)
        payload["envelopes"] = summarize(payload["records"])
        _write(payload, args.output_dir)
        print(
            json.dumps(
                {
                    **asdict(mutation),
                    "length": length,
                    "status": record["status"],
                    "stable": record["stable"],
                    "error": record.get("error"),
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
    print(f"Wrote {len(matrix)} records to {args.output_dir}")


if __name__ == "__main__":
    main()
