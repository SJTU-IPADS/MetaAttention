from __future__ import annotations

import pytest
import torch


_GPU_SKIP = pytest.mark.skip(reason="GPU tests require --run-gpu")
_BENCHMARK_SKIP = pytest.mark.skip(reason="benchmark tests require --run-benchmarks")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked gpu",
    )
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run tests marked benchmark or slow",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_gpu = config.getoption("--run-gpu")
    run_benchmarks = config.getoption("--run-benchmarks")
    for item in items:
        is_gpu = item.get_closest_marker("gpu") is not None
        is_benchmark = item.get_closest_marker("benchmark") is not None
        is_slow = item.get_closest_marker("slow") is not None
        if is_benchmark and not run_benchmarks:
            item.add_marker(_BENCHMARK_SKIP)
        elif is_slow and not (run_benchmarks or (is_gpu and run_gpu)):
            item.add_marker(_BENCHMARK_SKIP)
        elif is_gpu and not run_gpu and not (is_benchmark and run_benchmarks):
            item.add_marker(_GPU_SKIP)


@pytest.fixture
def gpu_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("GPU functional tests require an available CUDA or HIP device")
    return torch.device("cuda")


@pytest.fixture
def seed() -> int:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    return 0


@pytest.fixture
def require_h100(gpu_device: torch.device) -> torch.device:
    if torch.version.hip is not None:
        pytest.skip("H100 benchmark requires CUDA, not a HIP runtime")
    name = torch.cuda.get_device_name(gpu_device)
    capability = torch.cuda.get_device_capability(gpu_device)
    if "H100" not in name and capability < (9, 0):
        pytest.skip(
            f"H100 benchmark requires H100-compatible Hopper hardware; found {name} "
            f"with compute capability {capability}"
        )
    return gpu_device


@pytest.fixture
def require_mi250(gpu_device: torch.device) -> torch.device:
    if torch.version.hip is None:
        pytest.skip("MI250 benchmark requires a HIP runtime")
    name = torch.cuda.get_device_name(gpu_device)
    if "MI250" not in name.upper():
        pytest.skip(f"MI250 benchmark requires MI250-compatible hardware; found {name}")
    return gpu_device
