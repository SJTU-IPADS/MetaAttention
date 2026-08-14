from __future__ import annotations

import argparse
import json
import platform
import time

import torch
from tilelang.profiler import do_bench

from attn_engine import GDNEngine
from attn_engine.gdn_tilelang import _compile_gate_cumsum, _compile_kkt, _compile_state_output, _compile_wy


SHAPE = {"batch": 1, "query_heads": 4, "value_heads": 8, "length": 1024, "dim": 128}
WARMUP = 25
REPETITIONS = 100


def benchmark(device: torch.device) -> dict[str, object]:
    torch.cuda.set_device(device)
    device_index = torch.cuda.current_device()
    if torch.cuda.memory_allocated(device_index):
        raise RuntimeError("refusing to benchmark a GPU with existing PyTorch allocations")
    torch.cuda.reset_peak_memory_stats(device_index)
    batch, hq, hv, length, dim = SHAPE.values()
    query = torch.randn(batch, hq, length, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(batch, hv, length, dim, device=device, dtype=torch.bfloat16)
    gate = -torch.rand(batch, hv, length, device=device, dtype=torch.float32)
    beta = torch.rand(batch, hv, length, device=device, dtype=torch.float32)
    engine = GDNEngine(device)

    _compile_gate_cumsum.cache_clear()
    _compile_kkt.cache_clear()
    _compile_state_output.cache_clear()
    _compile_wy.cache_clear()
    started = time.perf_counter()
    engine(query, key, value, gate, beta)
    torch.cuda.synchronize(device_index)
    compile_ms = (time.perf_counter() - started) * 1e3
    gate_cumsum_kernel = _compile_gate_cumsum(batch, hv, length)
    kkt_kernel = _compile_kkt(batch, hq, hv, length)
    wy_kernel = _compile_wy(batch, hq, hv, length)
    state_kernel = _compile_state_output(batch, hq, hv, length, 128**-0.5, False)
    zero_state = torch.zeros(batch, hv, dim, dim, device=device, dtype=torch.float32)
    cumulative_gate = gate_cumsum_kernel(gate)
    key_inverse, value_inverse = kkt_kernel(key, beta, cumulative_gate)
    corrected_key, corrected_value = wy_kernel(
        key, value, beta, cumulative_gate, key_inverse, value_inverse
    )
    forward_stage_ms = {
        "gate_cumsum": do_bench(
            lambda: gate_cumsum_kernel(gate), warmup=WARMUP, rep=REPETITIONS
        ),
        "kkt": do_bench(
            lambda: kkt_kernel(key, beta, cumulative_gate),
            warmup=WARMUP,
            rep=REPETITIONS,
        ),
        "wy": do_bench(
            lambda: wy_kernel(
                key, value, beta, cumulative_gate, key_inverse, value_inverse
            ),
            warmup=WARMUP,
            rep=REPETITIONS,
        ),
        "state_output": do_bench(
            lambda: state_kernel(
                query,
                key,
                cumulative_gate,
                corrected_key,
                corrected_value,
                zero_state,
            ),
            warmup=WARMUP,
            rep=REPETITIONS,
        ),
    }
    end_to_end_ms = do_bench(
        lambda: engine(query, key, value, gate, beta),
        warmup=WARMUP,
        rep=REPETITIONS,
    )
    return {
        "shape": SHAPE,
        "dtype": {"query_key_value_output": "bfloat16", "gate_beta_state": "float32"},
        "warmup": WARMUP,
        "repetitions": REPETITIONS,
        "first_compile_ms": compile_ms,
        "forward_stage_ms": forward_stage_ms,
        "staged_forward_ms": sum(forward_stage_ms.values()),
        "end_to_end_ms": end_to_end_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
        "hardware": torch.cuda.get_device_name(device_index),
        "compute_capability": torch.cuda.get_device_capability(device_index),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = benchmark(torch.device(args.device))
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(payload + "\n")


if __name__ == "__main__":
    main()
