"""Interleaved public-call benchmark for unseen typed Stateful Operator compositions.

The benchmark compares the compiler-selected optimized candidate with the explicit
same-IR generic dense candidate. Each side gets ten paired measurement batches in
alternating order. Compilation and first-call code generation are excluded from
timing; peak allocation is measured on a warmed public call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from attn_engine import StateTuple, Target
from testing.stateful_composition_profiles import make_inputs, make_profile


KINDS = ("axis", "mixed", "injections", "rank-one")


def _call(compiled, values, initial):
    return compiled(
        **values,
        initial_state=StateTuple(("state",), (initial,)),
        return_final_state=True,
    )


def _timed_call(compiled, values, initial) -> tuple[float, float]:
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _call(compiled, values, initial)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - start_wall) * 1000.0
    return start.elapsed_time(end), wall_ms


def _measure(
    compiled,
    values,
    initial,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    with torch.no_grad():
        for _ in range(warmup):
            result = _call(compiled, values, initial)
            del result
        torch.cuda.synchronize()
        cuda_samples = []
        wall_samples = []
        for _ in range(repetitions):
            cuda_ms, wall_ms = _timed_call(compiled, values, initial)
            cuda_samples.append(cuda_ms)
            wall_samples.append(wall_ms)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        result = _call(compiled, values, initial)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del result
    return {
        "cuda_event_ms": statistics.median(cuda_samples),
        "wall_ms": statistics.median(wall_samples),
        "cuda_samples_ms": cuda_samples,
        "wall_samples_ms": wall_samples,
        "peak_incremental_allocated_bytes": peak - before,
    }


def _benchmark_kind(
    kind: str,
    *,
    batch: int,
    length: int,
    warmup: int,
    repetitions: int,
    batches: int,
) -> dict[str, object]:
    profile = make_profile(kind)
    optimized = profile.operator.compile(
        role_sizes=profile.role_sizes, target=Target(), mode="dense", optimized=True
    )
    generic = profile.operator.compile(
        role_sizes=profile.role_sizes, target=Target(), mode="dense", optimized=False
    )
    optimized_plan = profile.operator._plans[optimized.specialization]
    generic_plan = profile.operator._plans[generic.specialization]
    device = torch.device("cuda")
    values = make_inputs(profile, batch=batch, length=length, device=device)
    state = profile.operator.algorithm.states[0]
    initial = 0.01 * torch.randn(
        batch,
        profile.role_sizes[state.head_role],
        *(profile.role_sizes[role] for role in state.feature_roles),
        device=device,
    )
    measurements: dict[str, list[dict[str, object]]] = {"optimized": [], "generic": []}
    for batch_index in range(batches):
        order = (
            ("optimized", "generic")
            if batch_index % 2 == 0
            else ("generic", "optimized")
        )
        for label in order:
            compiled = optimized if label == "optimized" else generic
            measurements[label].append(
                _measure(
                    compiled,
                    values,
                    initial,
                    warmup=warmup,
                    repetitions=repetitions,
                )
            )
    optimized_cuda = [item["cuda_event_ms"] for item in measurements["optimized"]]
    generic_cuda = [item["cuda_event_ms"] for item in measurements["generic"]]
    optimized_wall = [item["wall_ms"] for item in measurements["optimized"]]
    generic_wall = [item["wall_ms"] for item in measurements["generic"]]
    optimized_memory = [
        item["peak_incremental_allocated_bytes"] for item in measurements["optimized"]
    ]
    generic_memory = [
        item["peak_incremental_allocated_bytes"] for item in measurements["generic"]
    ]
    return {
        "kind": kind,
        "batch": batch,
        "length": length,
        "device": torch.cuda.get_device_name(device),
        "dtype": "bfloat16 inputs, float32 gates/state",
        "contract": "one public call with final State",
        "optimized_runtime_family": optimized_plan.runtime_family,
        "optimized_lowerer": optimized_plan.plan.candidate.lowerer,
        "generic_runtime_family": generic_plan.runtime_family,
        "generic_lowerer": generic_plan.plan.candidate.lowerer,
        "measurement_batches": batches,
        "warmup": warmup,
        "repetitions_per_batch": repetitions,
        "optimized": measurements["optimized"],
        "generic": measurements["generic"],
        "paired_median": {
            "optimized_cuda_event_ms": statistics.median(optimized_cuda),
            "generic_cuda_event_ms": statistics.median(generic_cuda),
            "optimized_wall_ms": statistics.median(optimized_wall),
            "generic_wall_ms": statistics.median(generic_wall),
            "optimized_peak_incremental_allocated_bytes": statistics.median(
                optimized_memory
            ),
            "generic_peak_incremental_allocated_bytes": statistics.median(
                generic_memory
            ),
            "generic_over_optimized_cuda_ratio": statistics.median(generic_cuda)
            / statistics.median(optimized_cuda),
            "generic_over_optimized_wall_ratio": statistics.median(generic_wall)
            / statistics.median(optimized_wall),
            "optimized_over_generic_memory_ratio": statistics.median(optimized_memory)
            / statistics.median(generic_memory),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=129)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if (
        args.batch < 1
        or args.length < 1
        or args.warmup < 0
        or args.repetitions < 1
        or args.batches < 10
    ):
        raise SystemExit(
            "positive dimensions, nonnegative warmup, and at least ten batches are required"
        )
    torch.set_num_threads(1)
    records = [
        _benchmark_kind(
            kind,
            batch=args.batch,
            length=args.length,
            warmup=args.warmup,
            repetitions=args.repetitions,
            batches=args.batches,
        )
        for kind in args.kinds.split(",")
    ]
    payload = {
        "schema_version": "stateful-composition-benchmark/1",
        "comparison": "optimized typed lowerer versus generic dense lowerer for the same Algorithm IR",
        "ordering": "alternating per batch; paired batch medians",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
