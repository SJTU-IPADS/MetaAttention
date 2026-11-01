"""Compiler-owned candidate measurement and kernel configuration tuning.

This module deliberately has no dependency on the stateful planner.  The planner
supplies candidate objects and configuration tuples; this module only compares
already-lowered callables.  ``factory`` is the execution seam:

    factory(candidate, config, workload) -> callable

The returned callable is invoked with no arguments after it has been compiled
and warmed.  A workload is never inferred from a runtime tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # fcntl is available on the supported POSIX build, but keep imports safe.
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX hosts.
    fcntl = None  # type: ignore[assignment]

import torch


PROBE_BUDGET = 2
POLICY_VERSION = "stateful-workloads/provisional-h20-v1"


@dataclass(frozen=True)
class WorkloadPoint:
    sequence_count: int
    token_count: int
    distribution: str = "uniform"


@dataclass(frozen=True)
class CandidateSpec:
    """Optional adapter for callers that do not already have candidate objects."""

    identity: str
    configurations: tuple[tuple[tuple[str, int], ...], ...]
    static_cost: int = 0
    generic: bool = False


@dataclass(frozen=True)
class TuningFailure:
    candidate: str
    stage: str
    exception_type: str
    reason: str
    config: tuple[tuple[str, int], ...] | None = None
    workload_index: int | None = None


@dataclass(frozen=True)
class TuningResult:
    winner: Any
    winner_config: tuple[tuple[str, int], ...]
    scores: Mapping[str, float]
    latency_records: Mapping[str, tuple[float, ...]]
    compilation_timings: Mapping[str, tuple[float, ...]]
    failures: tuple[TuningFailure, ...] = ()
    cached: bool = False
    policy_version: str = POLICY_VERSION
    environment: Any = None
    config_measurements: Mapping[
        str, Mapping[tuple[tuple[str, int], ...], tuple[float, ...]]
    ] = ()

    @property
    def measurements(self) -> Mapping[str, tuple[float, ...]]:
        """Short compatibility spelling for consumers displaying benchmark data."""
        return self.latency_records


class GenericBaselineError(RuntimeError):
    """Raised when the mandatory generic candidate cannot produce evidence."""

    def __init__(self, failures: Sequence[TuningFailure]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(f.reason for f in self.failures) or "no valid measurement"
        super().__init__(f"generic stateful baseline failed measurement: {detail}")


class MeasurementCache:
    """Small JSON cache with process-safe locked, atomic successful publication.

    Cache entries are intentionally opaque dictionaries.  A record is published
    only after a complete successful tuning result exists; failed attempts never
    create negative entries.  The lock is held across the read/check/write
    sequence, so two compiler processes cannot overwrite a newer success.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._thread_lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._locked():
            if not self.path.exists():
                return None
            try:
                with self.path.open("r", encoding="utf-8") as stream:
                    value = json.load(stream)
            except (OSError, ValueError, TypeError):
                return None
            return value.get(key) if isinstance(value, dict) else None

    def publish(self, key: str, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            data: dict[str, Any] = {}
            if self.path.exists():
                try:
                    with self.path.open("r", encoding="utf-8") as stream:
                        loaded = json.load(stream)
                    if isinstance(loaded, dict):
                        data = loaded
                except (OSError, ValueError, TypeError):
                    data = {}
            data[key] = dict(value)
            fd, temporary = tempfile.mkstemp(
                prefix=self.path.name + ".", dir=self.path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, sort_keys=True, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    class _Lock:
        def __init__(self, owner: "MeasurementCache") -> None:
            self.owner = owner
            self.stream: Any = None

        def __enter__(self) -> "MeasurementCache._Lock":
            self.owner._thread_lock.acquire()
            if fcntl is not None:
                self.owner.lock_path.parent.mkdir(parents=True, exist_ok=True)
                self.stream = self.owner.lock_path.open("a+", encoding="utf-8")
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, *_: Any) -> None:
            if self.stream is not None:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
                self.stream.close()
            self.owner._thread_lock.release()

    def _locked(self) -> "MeasurementCache._Lock":
        return self._Lock(self)


def workload_lengths(workload: Any) -> tuple[int, ...]:
    """Expand a compiler-owned workload point into logical sequence lengths.

    ``long-tail`` is intentionally deterministic (one large sequence and a
    uniform remainder), while ``empty-containing`` always includes an empty
    sequence.  The sum is exactly ``token_count`` and no length is negative.
    """
    count = int(getattr(workload, "sequence_count"))
    tokens = int(getattr(workload, "token_count"))
    distribution = str(getattr(workload, "distribution", "uniform"))
    if count < 1 or tokens < 0:
        raise ValueError(
            "workload sequence_count must be positive and token_count non-negative"
        )
    if distribution == "empty-containing":
        if count == 1:
            return (0,)
        remaining = _uniform_lengths(count - 1, tokens)
        return (0, *remaining)
    if distribution == "long-tail":
        if count == 1:
            return (tokens,)
        large = max(0, tokens - max(1, tokens // max(2, count)))
        remainder = _uniform_lengths(count - 1, tokens - large)
        return (large, *remainder)
    return _uniform_lengths(count, tokens)


def _uniform_lengths(count: int, tokens: int) -> tuple[int, ...]:
    base, remainder = divmod(tokens, count)
    return tuple(base + (index < remainder) for index in range(count))


def _candidate_id(candidate: Any) -> str:
    value = getattr(candidate, "identity", candidate)
    if not isinstance(value, str):
        value = str(value)
    return value


def _is_generic(candidate: Any, generic: Any | None) -> bool:
    if generic is not None and candidate is generic:
        return True
    return bool(getattr(candidate, "generic", False)) or _candidate_id(candidate) in {
        "generic",
        "generic-dense-affine",
        "tilelang-generic",
    }


def _config(value: Any) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = tuple(value.items())
    result: list[tuple[str, int]] = []
    for name, amount in value:
        if (
            not isinstance(name, str)
            or isinstance(amount, bool)
            or not isinstance(amount, int)
        ):
            raise TypeError("kernel configurations must be tuple[(str, int)]")
        result.append((name, amount))
    return tuple(result)


def _canonical(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return str(value)


def _cache_key(
    candidate_ids: Sequence[str],
    configs: Mapping[str, Sequence[tuple[tuple[str, int], ...]]],
    workloads: Sequence[Any],
    policy_version: str,
    mode: str,
    environment: Any,
) -> str:
    identity = (
        tuple(candidate_ids),
        {key: tuple(configs[key]) for key in sorted(configs)},
        tuple(
            (
                _canonical(getattr(point, "sequence_count", None)),
                _canonical(getattr(point, "token_count", None)),
                _canonical(getattr(point, "distribution", None)),
            )
            for point in workloads
        ),
        policy_version,
        mode,
        _canonical(environment),
    )
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_get(cache: Any, key: str) -> Any | None:
    if cache is None:
        return None
    if hasattr(cache, "get"):
        return cache.get(key)
    if isinstance(cache, Mapping):
        return cache.get(key)
    return None


def _cache_publish(cache: Any, key: str, value: Mapping[str, Any]) -> None:
    if cache is None:
        return
    if hasattr(cache, "publish"):
        cache.publish(key, value)
    elif hasattr(cache, "__setitem__"):
        cache[key] = dict(value)


def _valid_latency(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("measurement must be finite and positive") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError("measurement must be finite and positive")
    return number


def _geomean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot score an empty workload")
    return math.exp(
        sum(math.log(_valid_latency(value)) for value in values) / len(values)
    )


def _cuda_measure(callable_: Callable[[], Any]) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA timing requires a CUDA device; inject timer for CPU tests"
        )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return _valid_latency(start.elapsed_time(end))


def _synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _invoke(
    factory: Callable[[Any, tuple[tuple[str, int], ...], Any], Callable[[], Any]],
    candidate: Any,
    config: tuple[tuple[str, int], ...],
    workload: Any,
    timer: Callable[[Callable[[], Any]], float] | None,
) -> tuple[float, float]:
    started = time.perf_counter()
    executable = factory(candidate, config, workload)
    # Compilation and allocation are deliberately outside the timed invocation.
    executable()
    _synchronize_if_cuda()
    compilation = time.perf_counter() - started
    measurement = _valid_latency((timer or _cuda_measure)(executable))
    return measurement, compilation


def _failure(
    candidate: Any,
    stage: str,
    error: BaseException,
    config: tuple[tuple[str, int], ...] | None = None,
    workload_index: int | None = None,
) -> TuningFailure:
    return TuningFailure(
        _candidate_id(candidate),
        stage,
        type(error).__name__,
        str(error) or repr(error),
        config,
        workload_index,
    )


def _serialize_result(result: TuningResult) -> dict[str, Any]:
    return {
        "candidate_ids": [_candidate_id(result.winner)],
        "winner": _candidate_id(result.winner),
        "winner_config": [[name, value] for name, value in result.winner_config],
        "scores": dict(result.scores),
        "latency_records": {
            key: list(value) for key, value in result.latency_records.items()
        },
        "compilation_timings": {
            key: list(value) for key, value in result.compilation_timings.items()
        },
        "policy_version": result.policy_version,
        "environment": _canonical(result.environment),
        "config_measurements": {
            candidate: {
                json.dumps(config): list(values) for config, values in configs.items()
            }
            for candidate, configs in result.config_measurements.items()
        },
    }


def _cached_result(
    value: Any,
    candidates: Sequence[Any],
    policy_version: str,
    environment: Any,
    workload_count: int,
    configs: Mapping[str, Sequence[tuple[tuple[str, int], ...]]],
) -> TuningResult | None:
    if not isinstance(value, Mapping) or value.get("policy_version") != policy_version:
        return None
    if _canonical(value.get("environment")) != _canonical(environment):
        return None
    candidate_map = {_candidate_id(candidate): candidate for candidate in candidates}
    winner_id = value.get("winner")
    winner = candidate_map.get(winner_id)
    if winner is None:
        return None
    try:
        winner_config = _config(value.get("winner_config", ()))
        expected_configs = configs[str(winner_id)]
        if winner_config not in expected_configs:
            return None
        records = {
            str(key): tuple(_valid_latency(item) for item in values)
            for key, values in dict(value["latency_records"]).items()
        }
        if not records or any(
            len(values) != workload_count for values in records.values()
        ):
            return None
        scores = {
            str(key): _valid_latency(item)
            for key, item in dict(value["scores"]).items()
        }
        if set(scores) != set(records):
            return None
        compile_times = {
            str(key): tuple(float(item) for item in values)
            for key, values in dict(value.get("compilation_timings", {})).items()
        }
    except (KeyError, TypeError, ValueError):
        return None
    return TuningResult(
        winner,
        winner_config,
        scores,
        records,
        compile_times,
        cached=True,
        policy_version=policy_version,
        environment=environment,
    )


def _workloads(policy: Any, mode: str) -> tuple[Any, ...]:
    if hasattr(policy, "points"):
        points = policy.points(mode)
    elif isinstance(policy, Mapping):
        points = policy[mode]
    else:
        points = policy
    points = tuple(points)
    if not points:
        raise ValueError("tuning workload policy must contain at least one point")
    for point in points:
        workload_lengths(
            point
        )  # validate compiler-owned policy before compiling anything.
    return points


def tune_candidates(
    candidates: Sequence[Any],
    config_spaces: Mapping[Any, Iterable[Any]],
    workload_policy: Any,
    factory: Callable[[Any, tuple[tuple[str, int], ...], Any], Callable[[], Any]],
    *,
    autotune: bool,
    mode: str = "dense",
    environment: Any = None,
    cache: Any = None,
    timer: Callable[[Callable[[], Any]], float] | None = None,
    generic_candidate: Any | None = None,
    static_costs: Mapping[Any, int] | None = None,
    probe_budget: int = PROBE_BUDGET,
) -> TuningResult:
    """Benchmark bounded candidates and tune the selected candidate only.

    ``timer`` is the explicit CPU test seam.  Omitting it always selects the
    CUDA-event timer, so production cannot silently rank candidates with a CPU
    wall clock.  ``autotune=False`` never invokes ``factory`` and only consumes
    a matching complete cache entry.
    """
    if not callable(factory):
        raise TypeError("factory must be callable")
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one lowered candidate is required")
    if probe_budget < 1:
        raise ValueError("probe_budget must be positive")
    workloads = _workloads(workload_policy, mode)
    identities = tuple(_candidate_id(candidate) for candidate in candidates)
    if len(set(identities)) != len(identities):
        raise ValueError("candidate identities must be unique")
    normalized_configs: dict[str, tuple[tuple[tuple[str, int], ...], ...]] = {}
    for candidate in candidates:
        identity = _candidate_id(candidate)
        try:
            raw = config_spaces.get(candidate, ())
        except TypeError:
            raw = ()
        if not raw:
            raw = config_spaces.get(identity, ())
        configs = tuple(dict.fromkeys(_config(item) for item in raw))
        if not configs:
            raise ValueError(f"candidate {identity!r} has no kernel configurations")
        normalized_configs[identity] = configs
    key = _cache_key(
        identities, normalized_configs, workloads, POLICY_VERSION, mode, environment
    )
    cached = _cached_result(
        _cache_get(cache, key),
        candidates,
        POLICY_VERSION,
        environment,
        len(workloads),
        normalized_configs,
    )
    if cached is not None:
        return cached
    costs: dict[str, int] = {}
    for candidate in candidates:
        identity = _candidate_id(candidate)
        provided = None
        if static_costs is not None:
            try:
                provided = static_costs.get(candidate)
            except TypeError:
                provided = None
            if provided is None:
                provided = static_costs.get(identity)
        costs[identity] = int(
            provided if provided is not None else getattr(candidate, "static_cost", 0)
        )
    # Measure the strongest feasible candidates; no representation is privileged.
    ranked = sorted(
        candidates, key=lambda item: (costs[_candidate_id(item)], _candidate_id(item))
    )
    measured_candidates = tuple(
        next(
            candidate
            for candidate in candidates
            if _candidate_id(candidate) == identity
        )
        for identity in dict.fromkeys(_candidate_id(item) for item in ranked[:3])
    )
    failures: list[TuningFailure] = []
    records_by_candidate: dict[str, tuple[float, ...]] = {}
    compile_by_candidate: dict[str, tuple[float, ...]] = {}
    all_config_records: dict[
        str, dict[tuple[tuple[str, int], ...], tuple[float, ...]]
    ] = {}

    if not autotune:
        winner = min(
            candidates,
            key=lambda item: (costs[_candidate_id(item)], _candidate_id(item)),
        )
        winner_id = _candidate_id(winner)
        winner_config = normalized_configs[winner_id][0]
        result = TuningResult(
            winner,
            winner_config,
            {},
            {},
            {},
            tuple(failures),
            False,
            POLICY_VERSION,
            environment,
            {},
        )
        return result

    # Equal small probe budget for every measured candidate.
    for candidate in measured_candidates:
        identity = _candidate_id(candidate)
        successful: list[
            tuple[tuple[tuple[str, int], ...], tuple[float, ...], tuple[float, ...]]
        ] = []
        for config in normalized_configs[identity][:probe_budget]:
            latencies: list[float] = []
            compile_times: list[float] = []
            config_failed = False
            for index, workload in enumerate(workloads):
                try:
                    latency, compile_time = _invoke(
                        factory, candidate, config, workload, timer
                    )
                    latencies.append(latency)
                    compile_times.append(compile_time)
                except (
                    BaseException
                ) as error:  # candidate failures are evidence, not process failures.
                    failures.append(_failure(candidate, "probe", error, config, index))
                    config_failed = True
                    break
            if not config_failed:
                successful.append((config, tuple(latencies), tuple(compile_times)))
                all_config_records.setdefault(identity, {})[config] = tuple(latencies)
        if successful:
            # One configuration must explain the candidate score.  Taking the
            # per-workload minimum across configurations is optimistic and could
            # select a configuration never measured end-to-end.
            chosen = min(successful, key=lambda item: (_geomean(item[1]), item[0]))
            records_by_candidate[identity] = chosen[1]
            compile_by_candidate[identity] = chosen[2]
    if not records_by_candidate:
        raise GenericBaselineError(tuple(failures))
    best_per_workload = tuple(
        min(values[index] for values in records_by_candidate.values())
        for index in range(len(workloads))
    )
    scores = {
        identity: math.exp(
            sum(
                math.log(values[index] / best_per_workload[index])
                for index in range(len(workloads))
            )
            / len(workloads)
        )
        for identity, values in records_by_candidate.items()
    }
    winner_id = min(
        scores, key=lambda identity: (scores[identity], costs[identity], identity)
    )
    winner = next(
        candidate for candidate in candidates if _candidate_id(candidate) == winner_id
    )

    # Full configuration search is intentionally restricted to the winner.
    full_records = all_config_records.setdefault(winner_id, {})
    full_compile_records: list[float] = list(compile_by_candidate.get(winner_id, ()))
    for config in normalized_configs[winner_id]:
        if config in full_records:
            continue
        latencies: list[float] = []
        compile_times: list[float] = []
        failed = False
        for index, workload in enumerate(workloads):
            try:
                latency, compile_time = _invoke(
                    factory, winner, config, workload, timer
                )
                latencies.append(latency)
                compile_times.append(compile_time)
            except BaseException as error:
                failures.append(_failure(winner, "autotune", error, config, index))
                failed = True
                break
        if not failed:
            full_records[config] = tuple(latencies)
            full_compile_records.extend(compile_times)
    if not full_records:
        raise RuntimeError(
            f"winning candidate {winner_id!r} has no valid full configuration"
        )
    winner_config, winner_latencies = min(
        full_records.items(), key=lambda item: (_geomean(item[1]), item[0])
    )
    records_by_candidate[winner_id] = winner_latencies
    compile_by_candidate[winner_id] = tuple(full_compile_records)
    if failures:
        details = "; ".join(
            f"{item.candidate} [{item.stage}/{item.exception_type}]: {item.reason}"
            for item in failures
        )
        warnings.warn(
            f"stateful tuning discarded failed attempts; selected {winner_id}: {details}",
            RuntimeWarning,
            stacklevel=2,
        )
    result = TuningResult(
        winner,
        winner_config,
        scores,
        records_by_candidate,
        compile_by_candidate,
        tuple(failures),
        False,
        POLICY_VERSION,
        environment,
        {key_: dict(value) for key_, value in all_config_records.items()},
    )
    _cache_publish(cache, key, _serialize_result(result))
    return result


# Deliberately simple aliases for integration code that calls the operation by
# its phase name rather than its implementation name.
benchmark_candidates = tune_candidates
tune = tune_candidates


__all__ = [
    "PROBE_BUDGET",
    "POLICY_VERSION",
    "WorkloadPoint",
    "CandidateSpec",
    "TuningFailure",
    "TuningResult",
    "GenericBaselineError",
    "MeasurementCache",
    "workload_lengths",
    "tune_candidates",
    "benchmark_candidates",
    "tune",
]
