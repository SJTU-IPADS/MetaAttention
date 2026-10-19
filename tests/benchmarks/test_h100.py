from __future__ import annotations

import csv
import json
import os
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

    artifact_dir = Path(os.environ.get("PR_MATRIX_ARTIFACT_DIR", str(tmp_path)))
    result_dir = artifact_dir / "h100-results"
    output_path = artifact_dir / "figure11_h100.pdf"
    csv_paths, pdf_path = run_figure11(result_dir, output_path)

    assert {path.name for path in csv_paths} == EXPECTED_CSV_NAMES
    assert {path.name for path in result_dir.glob("*.csv")} == EXPECTED_CSV_NAMES
    for path in csv_paths:
        assert path.is_file()
        assert path.stat().st_size > 0
        _assert_metaattention_csv(path)
    measurements = json.loads((result_dir / "measurements.json").read_text())
    meta = [record for record in measurements if record["method"] == "MetaAttention"]
    assert len(meta) == 62 * 2
    assert len({(record["case_id"], record["direction"]) for record in meta}) == 62 * 2
    for record in meta:
        if record["status"] in {"RESOURCE_LIMIT", "UNSUPPORTED"}:
            assert record["reason"]
            assert record["latency_ms"] is None
        elif record["status"] == "MEASURED_UNVALIDATED":
            assert record["latency_ms"] > 0
        else:
            assert record["direction"] == "backward"
            assert not record["require_grad"] or record["attn_type"] in {
                "retention_parallel",
                "mla_attn",
                "sparse_gqa",
            }
            assert record["status"] == "NOT_MEASURED"
    assert pdf_path == output_path
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 0
