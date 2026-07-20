from __future__ import annotations

import csv
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.h100,
]


EXPECTED_CSV_NAMES = {
    f"{name}_{direction}.csv"
    for name in (
        "deepseek",
        "llama",
        "vit",
        "dit",
        "retnet",
        "sigmoid_attn",
        "rfa",
        "mamba2",
        "yoco",
        "retnet_recur",
        "mla",
        "sparse_gqa",
    )
    for direction in ("fwd", "bwd")
}


def _assert_metaattention_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows and rows[0][0] == "Method"
    assert any(row and row[0] == "MetaAttention" for row in rows[1:])


def test_h100_runner_artifacts(tmp_path, require_h100):
    from benchmark.figure11 import run_figure11

    result_dir = tmp_path / "h100-results"
    output_path = tmp_path / "figure11_h100.pdf"
    csv_paths, pdf_path = run_figure11(result_dir, output_path)

    assert {path.name for path in csv_paths} == EXPECTED_CSV_NAMES
    assert {path.name for path in result_dir.glob("*.csv")} == EXPECTED_CSV_NAMES
    for path in csv_paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        _assert_metaattention_csv(path)
    assert pdf_path == output_path
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 0
