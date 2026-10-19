"""Public compiled scan-stage benchmarks versus optional FLA kernels.

Run from the repository root. Compilation, FLA layout binding and backward
forward-graph construction are excluded. Forward always uses inference_mode.
See docs/stateful-stage-benchmarks.md for the numerical and timing contracts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import hashlib
import subprocess
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.benchmark_fla_stateful import (  # noqa: E402
    _meta_failure_status,
    _physical_gpu_uuid,
    _visible_physical_indices,
    choose_idle_gpu,
    list_gpus,
)


@dataclass(frozen=True)
class Family:
    example: str
    factory: str
    fla_module: str
    chunk: str
    recurrent: str
    states: tuple[str, ...]
    kind: str | None = None


FAMILIES = {
    "preconditioned-kda": Family(
        "examples.preconditioned_delta",
        "preconditioned_delta",
        "fla.ops.precond_kda",
        "chunk_precond_kda",
        "fused_recurrent_precond_kda",
        ("atk", "memory"),
        "kda",
    ),
    "preconditioned-gdn": Family(
        "examples.preconditioned_delta",
        "preconditioned_delta",
        "fla.ops.precond_gated_delta_rule",
        "chunk_precond_gated_delta_rule",
        "fused_recurrent_precond_gated_delta_rule",
        ("atk", "memory"),
        "gdn",
    ),
    "lightnet": Family(
        "examples.lightnet_stable",
        "lightnet_stable",
        "fla.ops.lightnet",
        "chunk_lightnet",
        "fused_recurrent_lightnet",
        ("z", "memory"),
    ),
}
DTYPES = ("float32", "bfloat16", "float16")
FLA_SOURCE_ROOT = None
MEASUREMENT_CONTRACT = {
    "forward": "public invocation in torch.inference_mode; outputs and all finals returned",
    "backward": "autograd.grad of output and both finals; retained forward graph, construction excluded",
    "preprocessing": "Meta head-first inputs and FLA time-first contiguous inputs prebound; layout copies excluded on both sides",
    "included": "public validation, dispatch, stage bindings, internal preprocessing, allocations and launches",
    "excluded": "compile/JIT, external layout binding, oracle, forward graph setup for backward, no CUDA graph capture",
    "timing": "median CUDA-event and synchronized CPU wall around the same warmed public call; raw samples retained",
    "memory": "peak allocated and incremental allocated around one warmed call, before timing; backward graph is resident",
}


def matrix(families, seqlens, heads, dim, dtype):
    if not families or not seqlens or not heads:
        raise ValueError("matrix axes must not be empty")
    if set(families) - FAMILIES.keys():
        raise ValueError(f"families must be one of {tuple(FAMILIES)}")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if any(type(x) is not int or x < 1 for x in (*seqlens, *heads, dim)):
        raise ValueError("dimensions must be positive integers")
    return tuple((f, t, h) for f in families for t in seqlens for h in heads)


def build_operator(family, length, heads, dim, dtype):
    spec = FAMILIES[family]
    module = importlib.import_module(spec.example)
    if spec.kind:
        operator = getattr(module, spec.factory)(dtype=dtype, kind=spec.kind)
        roles = {
            module.KEY_HEADS: heads,
            module.VALUE_HEADS: heads,
            module.KEY: dim,
            module.VALUE: dim,
        }
    else:
        operator = getattr(module, spec.factory)(
            1, heads, length, dim, dim, dtype=dtype
        )
        roles = {module.HEADS: heads, module.KEY: dim, module.VALUE: dim}
    return operator, roles


def inputs_for(torch, family, length, heads, dim, dtype, device):
    """B1, equal K/V heads, FP32 gates and states, nonzero continuation caches."""
    vector = (1, heads, length, dim)
    scalar = vector[:-1]

    def rand(shape):
        return torch.rand(shape, device=device, dtype=torch.float32)

    def normal(shape):
        return torch.randn(shape, device=device, dtype=dtype) * 0.3

    inputs = {"query": normal(vector), "value": normal(vector)}
    memory = torch.randn(1, heads, dim, dim, device=device) * 0.1
    if FAMILIES[family].kind:
        inputs.update(
            key=normal(vector),
            log_atk_decay=-rand(scalar) * 0.2,
            beta_atk=rand(scalar) * 0.4,
            center=torch.full(scalar, -0.2, device=device, dtype=torch.float32),
            scale=torch.full(scalar, dim**-0.5, device=device, dtype=torch.float32),
            beta=rand(scalar) * 0.3,
            log_decay=-rand(vector if FAMILIES[family].kind == "kda" else scalar) * 0.1,
        )
        initial = (rand((1, heads, dim)) * 0.1, memory)
    else:
        inputs["logits"] = normal(vector)
        # A finite normalizer verifies genuine continuation, including T1.
        initial = (torch.randn(1, heads, dim, device=device), memory)
    return {n: t.contiguous() for n, t in inputs.items()}, initial


def oracle(torch, family, inputs, initial):
    """Independent authored-order FP32 token recurrence (also differentiable)."""
    import math

    q, v = inputs["query"].float(), inputs["value"].float()
    a, s = (t.float() for t in initial)
    kind = FAMILIES[family].kind
    if kind:
        k = inputs["key"].float()
        if kind == "gdn":
            q, k = (torch.nn.functional.normalize(t, dim=-1, eps=1e-12) for t in (q, k))
    outputs = []
    for t in range(q.shape[2]):
        if kind:
            kt = k[:, :, t]
            a = (
                inputs["log_atk_decay"][:, :, t].exp()[..., None] * a
                + inputs["beta_atk"][:, :, t, None] * kt.square()
            )
            r = (a + 1e-6).log() - inputs["center"][:, :, t, None]
            write = kt * torch.exp(-math.log(1.5) * r / (1 + r.abs()))
            decay = inputs["log_decay"][:, :, t].exp()
            s = s * (decay[..., None] if kind == "kda" else decay[..., None, None])
            error = v[:, :, t] - (kt[..., None] * s).sum(-2)
            s = (
                s
                + write[..., None]
                * (inputs["beta"][:, :, t, None] * error)[..., None, :]
            )
            qt = q[:, :, t] * inputs["scale"][:, :, t, None]
        else:
            previous = a
            a = torch.logaddexp(a, inputs["logits"][:, :, t].float())
            write = (inputs["logits"][:, :, t].float() - a).exp()
            gate = torch.where(torch.isneginf(previous), 0.0, previous - a)
            s = gate.exp()[..., None] * s + write[..., None] * v[:, :, t, None, :]
            qt = q[:, :, t] * q.shape[-1] ** -0.5
        outputs.append((qt[..., None] * s).sum(-2))
    return {
        "output": torch.stack(outputs, 2).to(inputs["value"].dtype),
        FAMILIES[family].states[0]: a,
        "memory": s,
    }


def meta_call(compiled, family, inputs, initial):
    from attn_engine import StateTuple

    state = StateTuple(FAMILIES[family].states, initial)

    def run():
        result = compiled(**inputs, initial_state=state, return_final_state=True)
        return {**result.outputs.as_dict(), **result.final_state.as_dict()}

    return run


def fla_call(torch, family, variant, inputs, initial, *, prebound=False):
    """Prebind layout only; all FLA gate/normalization operations remain timed."""
    spec = FAMILIES[family]
    if FLA_SOURCE_ROOT is not None and not (FLA_SOURCE_ROOT / "fla").is_dir():
        raise ImportError(f"FLA source directory does not exist: {FLA_SOURCE_ROOT}")
    module = importlib.import_module(spec.fla_module)
    if FLA_SOURCE_ROOT is not None:
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(FLA_SOURCE_ROOT):
            raise ImportError(f"FLA resolved outside requested source: {module_path}")
    function = getattr(module, getattr(spec, variant))
    native = (
        inputs
        if prebound
        else {n: t.transpose(1, 2).contiguous() for n, t in inputs.items()}
    )
    common = dict(
        q=native["query"],
        v=native["value"],
        scale=inputs["query"].shape[-1] ** -0.5,
        output_final_state=True,
    )
    if spec.kind:
        common.update(
            k=native["key"],
            g=native["log_decay"],
            g_atk=native["log_atk_decay"],
            beta_atk=native["beta_atk"],
            beta=native["beta"],
            initial_state=initial[1],
            initial_A_state=initial[0],
        )
        if spec.kind == "gdn" or variant == "recurrent":
            common["use_qk_l2norm_in_kernel"] = spec.kind == "gdn"
    else:
        common.update(
            k=native["logits"], initial_state=(initial[1], initial[0].unsqueeze(1))
        )

    def run():
        result = function(**common)
        if spec.kind:
            o, memory, auxiliary = result
        else:
            o, (memory, auxiliary) = result
            auxiliary = auxiliary.squeeze(1)
        # A view, not a copy; canonical layout for validation and the loss.
        return {
            "output": o.transpose(1, 2),
            spec.states[0]: auxiliary,
            "memory": memory,
        }

    return run


def correctness(torch, actual, expected, *, rtol=0.02, atol=0.005):
    fields = {}
    valid = actual.keys() == expected.keys()
    for name in sorted(actual.keys() | expected.keys()):
        a, e = actual.get(name), expected.get(name)
        if name in actual and name in expected and a is None and e is None:
            fields[name] = {"status": "VALID", "reason": "both unused"}
            continue
        if a is None or e is None or a.shape != e.shape:
            fields[name] = {"status": "INVALID", "reason": "missing or shape mismatch"}
            valid = False
            continue
        a, e = a.detach().cpu().float(), e.detach().cpu().float()
        delta = (a - e).abs()
        passed = bool(
            torch.isfinite(a).all() and torch.allclose(a, e, rtol=rtol, atol=atol)
        )
        maximum, rms = delta.max().item(), delta.square().mean().sqrt().item()
        fields[name] = {
            "status": "VALID" if passed else "INVALID",
            "max_abs": maximum if math.isfinite(maximum) else None,
            "rms": rms if math.isfinite(rms) else None,
            "nonfinite_elements": int((~torch.isfinite(a)).sum()),
        }
        valid &= passed
    return {
        "status": "VALID" if valid else "INVALID",
        "rtol": rtol,
        "atol": atol,
        "fields": fields,
    }


def measure(torch, call, warmup, reps):
    # A mandatory untimed prime excludes lazy backward JIT even at warmup=0.
    for _ in range(max(1, warmup)):
        call()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    value = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del value
    samples, wall = [], []
    for _ in range(reps):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        torch.cuda.synchronize()
        then = time.perf_counter()
        start.record()
        value = call()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - then) * 1000)
        samples.append(start.elapsed_time(end))
        del value
    return {
        "public_cuda_event_ms": statistics.median(samples),
        "cpu_wall_ms": statistics.median(wall),
        "cuda_samples_ms": samples,
        "wall_samples_ms": wall,
        "resident_allocated_bytes": resident,
        "peak_allocated_bytes": peak,
        "peak_incremental_bytes": max(0, peak - resident),
    }


def child_lowerers(compiled):
    result = {}
    for name, child in compiled.children.items():
        plan = getattr(child.operator, "_plans", {}).get(child.specialization)
        result[name] = {
            "runtime_family": getattr(
                plan, "runtime_family", getattr(child, "runtime_family", None)
            ),
            "selected_lowerer": plan.plan.candidate.lowerer if plan else None,
        }
    return result


def gradient_call(torch, forward, tensors):
    values = forward()
    # Fixed unit cotangents for every output and final; no loss assembly timed.
    targets = tuple(values.values())
    weights = tuple(torch.ones_like(t) for t in targets)

    def run():
        return torch.autograd.grad(
            targets, tensors, grad_outputs=weights, retain_graph=True, allow_unused=True
        )

    return run


def _failure(error, meta=False):
    return {
        "status": _meta_failure_status(error) if meta else "FAILED",
        "error_type": type(error).__name__,
        "error": str(error),
        "details": error.as_dict() if hasattr(error, "as_dict") else {},
    }


def run_case(
    torch,
    family,
    length,
    heads,
    dim,
    dtype,
    workspace,
    warmup,
    reps,
    backward,
    checkpoint=None,
):
    from attn_engine import Target

    record = {
        "family": family,
        "batch": 1,
        "heads": heads,
        "dim_key": dim,
        "dim_value": dim,
        "seqlen": length,
        "workload": "decode" if length == 1 else "prefill",
        "workspace_limit_bytes": workspace,
        "dtype": str(dtype),
        "backends": {},
    }
    torch.manual_seed(42)
    inputs, initial = inputs_for(torch, family, length, heads, dim, dtype, "cuda")
    # CPU oracle is untimed, independent of any engine fallback or FLA kernel.
    cpu_inputs = {n: t.detach().cpu() for n, t in inputs.items()}
    cpu_initial = tuple(t.detach().cpu() for t in initial)
    with torch.no_grad():
        expected = oracle(torch, family, cpu_inputs, cpu_initial)
    compiled = None
    try:
        operator, roles = build_operator(family, length, heads, dim, dtype)
        compiled = operator.compile(
            role_sizes=roles,
            target=Target(workspace_bytes=workspace),
            mode="dense",
            backward=False,
        )
        record["selected_child_lowerers"] = child_lowerers(compiled)
        record["staged_workspace_estimate_bytes"] = compiled.workspace_bytes(
            length, sequence_count=1, sequence_length=length
        )
    except Exception as error:
        record["backends"]["meta_forward"] = _failure(error, True)

    if checkpoint:
        checkpoint(record)
    meta_actual = None
    for backend in ("meta", "fla_chunk", "fla_recurrent"):
        key = backend + "_forward"
        if backend == "meta" and key in record["backends"]:
            continue
        print(
            json.dumps({"backend_start": key, "family": family, "T": length}),
            flush=True,
        )
        try:
            call = (
                meta_call(compiled, family, inputs, initial)
                if backend == "meta"
                else fla_call(torch, family, backend.split("_")[1], inputs, initial)
            )
            with torch.inference_mode():
                actual_gpu = call()
                actual = {n: t.detach().cpu() for n, t in actual_gpu.items()}
                del actual_gpu
                check = correctness(torch, actual, expected)
                pair = (
                    correctness(torch, actual, meta_actual)
                    if meta_actual is not None and backend != "meta"
                    else None
                )
                timing = measure(torch, call, warmup, reps)
            entry = {
                "status": "OK" if check["status"] == "VALID" else "NUMERICAL_MISMATCH",
                "oracle_correctness": check,
                "measurement": timing,
            }
            if backend == "meta":
                meta_actual = {n: t.detach().cpu() for n, t in actual.items()}
            elif pair is not None:
                entry["meta_pair_difference"] = pair
            if family == "preconditioned-kda" and backend == "fla_chunk":
                entry["baseline_semantic_note"] = (
                    "This FLA public chunk API hardcodes q/k L2 normalization; "
                    "the authored KDA and FLA recurrent API use original q/k. "
                    "Raw API result retained, not silently substituted or renormalized."
                )
            record["backends"][key] = entry
            del actual, call
        except Exception as error:
            record["backends"][key] = _failure(error, backend == "meta")
        if checkpoint:
            checkpoint(record)

    if backward:
        # Every backend receives independent leaves; FLA layout copies do not
        # occur inside the VJP interval. Meta includes its public autograd path.
        for backend in ("meta", "fla_chunk"):
            print(
                json.dumps(
                    {
                        "backend_start": backend + "_backward",
                        "family": family,
                        "T": length,
                    }
                ),
                flush=True,
            )
            try:
                grad_inputs = {
                    n: (t if backend == "meta" else t.transpose(1, 2))
                    .detach()
                    .contiguous()
                    .clone()
                    .requires_grad_(n not in ("center", "scale"))
                    for n, t in inputs.items()
                }
                grad_initial = tuple(
                    t.detach().clone().requires_grad_() for t in initial
                )
                gradient_names = tuple(
                    n for n in grad_inputs if n not in ("center", "scale")
                )
                tensors = (*(grad_inputs[n] for n in gradient_names), *grad_initial)
                if backend == "meta":
                    op, roles = build_operator(family, length, heads, dim, dtype)
                    grad_compiled = op.compile(
                        role_sizes=roles,
                        target=Target(workspace_bytes=workspace),
                        mode="dense",
                        backward=True,
                    )
                    record["backward_child_lowerers"] = child_lowerers(grad_compiled)
                    record["backward_staged_workspace_estimate_bytes"] = (
                        grad_compiled.workspace_bytes(
                            length, sequence_count=1, sequence_length=length
                        )
                    )
                    forward = meta_call(
                        grad_compiled, family, grad_inputs, grad_initial
                    )
                else:
                    forward = fla_call(
                        torch, family, "chunk", grad_inputs, grad_initial, prebound=True
                    )
                forward_check = correctness(torch, forward(), expected)
                call = gradient_call(torch, forward, tensors)
                entry = {
                    "status": ("OK" if length <= 64 else "MEASURED_UNVALIDATED")
                    if forward_check["status"] == "VALID"
                    else "NUMERICAL_MISMATCH",
                    "forward_oracle_correctness": forward_check,
                    "measurement": measure(torch, call, warmup, reps),
                    "gradient_oracle": {
                        "status": "NOT_CHECKED",
                        "reason": "full gradient oracle limited to T<=64",
                    },
                }
                if length <= 64:
                    ref_inputs = {
                        n: t.detach().cpu().requires_grad_() for n, t in inputs.items()
                    }
                    ref_initial = tuple(
                        t.detach().cpu().requires_grad_() for t in initial
                    )
                    ref_call = gradient_call(
                        torch,
                        lambda: oracle(torch, family, ref_inputs, ref_initial),
                        (*(ref_inputs[n] for n in gradient_names), *ref_initial),
                    )
                    actual_grad, expected_grad = call(), ref_call()
                    if backend != "meta":
                        actual_grad = tuple(
                            g.transpose(1, 2)
                            if g is not None and i < len(gradient_names)
                            else g
                            for i, g in enumerate(actual_grad)
                        )
                    names = (*gradient_names, *FAMILIES[family].states)
                    check = correctness(
                        torch,
                        dict(zip(names, actual_grad)),
                        dict(zip(names, expected_grad)),
                        rtol=0.04,
                        atol=0.02,
                    )
                    entry["gradient_oracle"] = check
                    if check["status"] != "VALID":
                        entry["status"] = "NUMERICAL_MISMATCH"
                    del actual_grad, expected_grad, ref_call, ref_inputs, ref_initial
                entry["gradient_inputs"] = [*gradient_names, *FAMILIES[family].states]
                entry["excluded_gradients"] = [
                    n for n in grad_inputs if n not in gradient_names
                ]
                record["backends"][backend + "_backward"] = entry
                del call, forward, tensors, grad_inputs, grad_initial
            except Exception as error:
                record["backends"][backend + "_backward"] = _failure(
                    error, backend == "meta"
                )
            if checkpoint:
                checkpoint(record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", "--family", default=",".join(FAMILIES))
    parser.add_argument("--seqlens", type=int, nargs="+", default=[1, 64, 2048, 8192])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, nargs="+", default=[1])
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--workspace", type=int, default=4 << 30, help="workspace cap in bytes"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=None, help="physical idle GPU ID")
    parser.add_argument(
        "--fla-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "flash-linear-attention",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/stages.json")
    )
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument(
        "--case-timeout",
        type=int,
        default=900,
        help="per-workload child deadline in seconds",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.families = args.families.split(",")
    try:
        matrix(args.families, args.seqlens, args.heads, args.dim, args.dtype)
    except ValueError as error:
        parser.error(str(error))
    if args.workspace < 1 or args.warmup < 0 or args.reps < 1 or args.case_timeout < 1:
        parser.error("workspace/reps must be positive and warmup nonnegative")
    return args


def provenance(torch, fla_path):
    def revision(path):
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None

    digest = hashlib.sha256()
    for directory in (ROOT / "attention_engine" / "attn_engine", ROOT / "examples"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(
                str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes()
            )
    digest.update(Path(__file__).read_bytes())
    digest.update((ROOT / "testing" / "benchmark_fla_stateful.py").read_bytes())
    fla_digest = hashlib.sha256()
    for path in sorted((fla_path / "fla").rglob("*.py")):
        fla_digest.update(
            str(path.relative_to(fla_path)).encode() + b"\0" + path.read_bytes()
        )
    return {
        "git_commit": revision(ROOT),
        "source_sha256": digest.hexdigest(),
        "fla_git_commit": revision(fla_path),
        "fla_source_sha256": fla_digest.hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def isolated_case(args, gpu, family, length, heads):
    identity = {"family": family, "seqlen": length, "heads": heads}
    try:
        snapshot = list_gpus()
        choose_idle_gpu(snapshot, requested=gpu)
    except RuntimeError as error:
        keys = ["meta_forward", "fla_chunk_forward", "fla_recurrent_forward"]
        if not args.no_backward:
            keys += ["meta_backward", "fla_chunk_backward"]
        return {
            **identity,
            "status": "GPU_BUSY_REJECTED",
            "error": str(error),
            "backends": {key: {"status": "GPU_BUSY_REJECTED"} for key in keys},
        }
    with tempfile.TemporaryDirectory(prefix="fla-stage-") as temporary:
        result_path = Path(temporary) / "record.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--family",
            family,
            "--seqlens",
            str(length),
            "--heads",
            str(heads),
            "--dim",
            str(args.dim),
            "--dtype",
            args.dtype,
            "--workspace",
            str(args.workspace),
            "--warmup",
            str(args.warmup),
            "--reps",
            str(args.reps),
            "--fla-path",
            str(args.fla_path.resolve()),
            "--output",
            str(result_path),
        ]
        if args.no_backward:
            command.append("--no-backward")
        failure = None
        try:
            completed = subprocess.run(command, timeout=args.case_timeout, check=False)
            if completed.returncode:
                failure = {"status": "CHILD_FAILED", "returncode": completed.returncode}
        except subprocess.TimeoutExpired:
            failure = {"status": "CHILD_TIMEOUT", "timeout_seconds": args.case_timeout}
        record = (
            json.loads(result_path.read_text()) if result_path.exists() else identity
        )
        if failure is None and not result_path.exists():
            failure = {"status": "CHILD_FAILED", "error": "worker returned no JSON"}
        if failure:
            record.update(failure)
            # Preserve completed backends and explicitly account for unfinished ones.
            expected = ["meta_forward", "fla_chunk_forward", "fla_recurrent_forward"]
            if not args.no_backward:
                expected += ["meta_backward", "fla_chunk_backward"]
            for key in expected:
                record.setdefault("backends", {}).setdefault(key, dict(failure))
        record["gpu_snapshot_before"] = [asdict(g) for g in snapshot if g.index == gpu]
        try:
            after = next(g for g in list_gpus() if g.index == gpu)
            record["gpu_snapshot_after"] = asdict(after)
            if after.compute_pids:
                record["timing_validity"] = "CONTENDED_EXTERNAL_PROCESS"
            else:
                record["timing_validity"] = "NO_EXTERNAL_PROCESS_OBSERVED_AT_BOUNDARIES"
        except (RuntimeError, StopIteration) as error:
            record["timing_validity"] = "GPU_QUERY_FAILED"
            record["gpu_query_error"] = str(error)
        return record


def main(argv=None):
    global FLA_SOURCE_ROOT
    args = parse_args(argv)
    FLA_SOURCE_ROOT = args.fla_path.resolve()
    # Workers inherit the physical mask; only the controller selects a GPU.
    selected = None
    if not args.worker:
        selected = choose_idle_gpu(
            list_gpus(), requested=args.gpu, allowed_indices=_visible_physical_indices()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = _physical_gpu_uuid(selected.index)
    sys.path.insert(0, str(args.fla_path.resolve()))
    import torch

    torch.set_num_threads(1)
    if os.environ.get("STATEFUL_OPERATOR_TEST_CPU"):
        raise RuntimeError(
            "remove STATEFUL_OPERATOR_TEST_CPU: refusing test fallback in GPU benchmark"
        )
    if args.worker:
        family, length, heads = matrix(
            args.families, args.seqlens, args.heads, args.dim, args.dtype
        )[0]
        args.output.parent.mkdir(parents=True, exist_ok=True)

        worker_provenance = provenance(torch, args.fla_path)

        def save(record):
            record["provenance"] = worker_provenance
            args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")

        try:
            record = run_case(
                torch,
                family,
                length,
                heads,
                args.dim,
                getattr(torch, args.dtype),
                args.workspace,
                args.warmup,
                args.reps,
                not args.no_backward,
                checkpoint=save,
            )
        except Exception as error:
            record = {
                "family": family,
                "seqlen": length,
                "heads": heads,
                "status": "HARNESS_ERROR",
                "error": repr(error),
            }
        save(record)
        return
    manifest = {
        "schema_version": 1,
        "selected_gpu": asdict(selected),
        "measurement_contract": MEASUREMENT_CONTRACT,
        "warmup": args.warmup,
        "reps": args.reps,
        "exploratory": args.reps < 20,
        "case_timeout_seconds": args.case_timeout,
        "isolation": "fresh child process per workload; backend checkpoints retained on child failure",
        "fla_path": str(args.fla_path.resolve()),
        "provenance": provenance(torch, args.fla_path),
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for family, length, heads in matrix(
        args.families, args.seqlens, args.heads, args.dim, args.dtype
    ):
        print(json.dumps({"starting": family, "T": length, "H": heads}), flush=True)
        record = isolated_case(args, selected.index, family, length, heads)
        manifest["records"].append(record)
        args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    print(
        json.dumps({"output": str(args.output), "records": len(manifest["records"])}),
        flush=True,
    )


if __name__ == "__main__":
    # Also support python testing/benchmark_fla_stages.py without installation.
    main()
