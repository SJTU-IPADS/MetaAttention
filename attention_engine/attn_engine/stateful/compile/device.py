"""Hardware profiles and schedule candidates for Stateful compilation.

The Stateful IR and mathematical plans are target-independent.  This module owns
hardware facts and the small, lowering-facing default schedule space.  It does
not select a winner; measurement remains the responsibility of ``tuning.py``.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceProfile:
    """Immutable hardware contract used to constrain compiled schedules."""

    identity: str
    device_type: str
    product_family: str
    capability: tuple[int, int]
    max_threads_per_block: int
    shared_memory_per_block: int
    register_limit_per_block: int
    default_threads: tuple[int, ...]
    default_block_tokens: tuple[int, ...]

    @property
    def target_identity(self) -> tuple[Any, ...]:
        return (
            self.identity,
            self.device_type,
            self.product_family,
            self.capability,
            self.max_threads_per_block,
            self.shared_memory_per_block,
            self.register_limit_per_block,
        )


# The values are deliberately the existing Stateful defaults.  They describe a
# legal candidate domain, not a promise that one configuration wins every shape.
_H20 = DeviceProfile(
    identity="nvidia-h20-sm90",
    device_type="cuda",
    product_family="H20",
    capability=(9, 0),
    max_threads_per_block=1024,
    shared_memory_per_block=99_264,
    register_limit_per_block=65_536,
    default_threads=(128, 256),
    default_block_tokens=(128, 64, 256),
)

_CPU = DeviceProfile(
    identity="cpu-generic",
    device_type="cpu",
    product_family="CPU",
    capability=(0, 0),
    max_threads_per_block=1024,
    shared_memory_per_block=99_264,
    register_limit_per_block=65_536,
    default_threads=(128, 256),
    default_block_tokens=(128, 64, 256),
)

DEFAULT_DEVICE_PROFILE = _H20


def resolve_device_profile(target: Any) -> DeviceProfile:
    """Resolve a deterministic profile from the public target contract.

    Compile-time resolution intentionally does not query CUDA.  Runtime target
    validation still checks the actual product and capability before execution.
    """
    device_type = getattr(target, "device_type", "cuda")
    product_family = str(getattr(target, "product_family", "H20"))
    capability = tuple(getattr(target, "capability", (9, 0)))
    if (
        device_type == "cuda"
        and product_family.upper() == "H20"
        and capability == (9, 0)
    ):
        return _H20
    if device_type == "cpu":
        return _CPU
    return DeviceProfile(
        identity=f"{device_type}-{product_family.lower()}-sm{capability[0]}{capability[1]}",
        device_type=device_type,
        product_family=product_family,
        capability=capability,
        max_threads_per_block=1024,
        shared_memory_per_block=99_264,
        register_limit_per_block=65_536,
        default_threads=(128, 256),
        default_block_tokens=(128, 64, 256),
    )


@dataclass(frozen=True)
class ScheduleConfig:
    """One lowering configuration that can be measured and cached."""

    values: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.values)
        if names != tuple(sorted(set(names))):
            raise ValueError("schedule configuration names must be unique and sorted")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for _, value in self.values
        ):
            raise ValueError("schedule configuration values must be positive integers")

    @property
    def identity(self) -> tuple[tuple[str, int], ...]:
        return self.values


def schedule_configurations(
    candidate: Any, profile: DeviceProfile
) -> tuple[ScheduleConfig, ...]:
    """Return the legal first-generation schedule domain for one candidate."""
    if getattr(candidate, "lowerer", None) == "tilelang-elementwise":
        return tuple(
            ScheduleConfig((("block_tokens", size),))
            for size in profile.default_block_tokens
        )
    return tuple(
        ScheduleConfig((("threads", size),))
        for size in profile.default_threads
        if size <= profile.max_threads_per_block
    )
