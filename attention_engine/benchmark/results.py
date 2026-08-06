from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


Metric = tuple[float | None, float | None] | None
BenchData = Sequence[tuple[str, Mapping[str, Metric]]]


def dump_bench_result(
    name: str,
    data: BenchData,
    result_dir: str | Path,
) -> tuple[Path, Path]:
    """Write benchmark metrics as the legacy forward/backward CSV pair.

    Configuration columns retain first-seen order. Missing and explicit
    ``None`` metrics are emitted as empty CSV cells, matching pandas' output
    from the original scripts without requiring pandas at import time.
    """

    result_dir = Path(result_dir)
    fwd_path = result_dir / f"{name}_fwd.csv"
    bwd_path = result_dir / f"{name}_bwd.csv"
    if not data:
        return fwd_path, bwd_path

    config_order: list[str] = []
    fwd_values: dict[str, dict[str, float | None]] = {}
    bwd_values: dict[str, dict[str, float | None]] = {}
    method_order: list[str] = []

    for config_name, metrics in data:
        if config_name not in config_order:
            config_order.append(config_name)
        for method_name, values in metrics.items():
            if method_name not in method_order:
                method_order.append(method_name)
            if values is None:
                fwd_value = bwd_value = None
            else:
                fwd_value = values[0] if len(values) >= 1 else None
                bwd_value = values[1] if len(values) >= 2 else None
            fwd_values.setdefault(method_name, {})[config_name] = fwd_value
            bwd_values.setdefault(method_name, {})[config_name] = bwd_value

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_metric_csv(fwd_path, method_order, config_order, fwd_values)
    _write_metric_csv(bwd_path, method_order, config_order, bwd_values)
    return fwd_path, bwd_path


def _write_metric_csv(
    path: Path,
    method_order: Iterable[str],
    config_order: Iterable[str],
    values: Mapping[str, Mapping[str, float | None]],
) -> None:
    configs = list(config_order)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Method", *(config.replace("\n", " ") for config in configs)])
        for method in method_order:
            row = [method]
            method_values = values.get(method, {})
            row.extend(
                "" if method_values.get(config) is None else method_values[config]
                for config in configs
            )
            writer.writerow(row)


def load_csv_data(
    filename: str | Path,
    exclude_kv: bool = False,
) -> tuple[list[str], list[tuple[str, list[float]]]]:
    """Load plotting data with the legacy missing/invalid-value behavior."""

    filename = Path(filename)
    if not filename.exists():
        print(f"Warning: File {filename} not found.")
        return [], []

    providers: list[str] = []
    times_data: list[tuple[str, list[float]]] = []
    with filename.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            headers = next(reader)
        except StopIteration:
            return [], []

        target_indices: list[int] = []
        for index, column_name in enumerate(headers):
            if index == 0:
                continue
            if exclude_kv and "KV" in column_name:
                continue
            target_indices.append(index)
            providers.append(column_name.replace(" ", "\n"))

        for row in reader:
            if not row:
                continue
            method_name = row[0]
            row_values: list[float] = []
            for index in target_indices:
                value_string = row[index] if index < len(row) else ""
                try:
                    value = float(value_string)
                except ValueError:
                    value = 0.0
                row_values.append(value)
            times_data.append((method_name, row_values))

    return providers, times_data


__all__ = ["dump_bench_result", "load_csv_data"]
