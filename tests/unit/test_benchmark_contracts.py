from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from benchmark import dump_bench_result, h100_cases, load_csv_data, mi250_cases
from benchmark import figure11, figure14
from benchmark import plot_fig_h100
from benchmark.plot_fig_mi250 import plot_figure14


pytestmark = [pytest.mark.unit, pytest.mark.io]


H100_FILES = (
    "deepseek",
    "llama",
    "dit",
    "sigmoid_attn",
    "vit",
    "retnet",
    "mamba2",
    "rfa",
    "yoco",
    "retnet_recur",
    "mla",
    "sparse_gqa",
)
MI250_FILES = ("deepseek", "vit", "mamba2", "retnet_recur", "mla")


def _write_synthetic_csvs(result_dir: Path, names: tuple[str, ...]) -> None:
    result_dir.mkdir()
    for name in names:
        for direction in ("fwd", "bwd"):
            path = result_dir / f"{name}_{direction}.csv"
            path.write_text(
                "Method,BS1 S2048,BS8 S4096,BS8S1 KV8192\n"
                "MetaAttention,1.0,1.1,1.2\n"
                "Torch Inductor,2.0,2.1,2.2\n",
                encoding="utf-8",
            )


def _assert_metaattention_rows(paths: tuple[Path, ...]) -> None:
    for path in paths:
        rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
        assert rows
        assert any(row and row[0] == "MetaAttention" for row in rows[1:])


def test_benchmark_case_matrices_are_exact():
    h100 = h100_cases()
    mi250 = mi250_cases()
    assert len(h100) == 62
    assert len(mi250) == 25
    assert Counter(case.dataset for case in h100) == Counter(
        {
            "deepseek": 12,
            "llama": 6,
            "vit": 6,
            "dit": 6,
            "retnet": 4,
            "sigmoid_attn": 6,
            "rfa": 3,
            "mamba2": 6,
            "yoco": 3,
            "retnet_recur": 4,
            "mla": 3,
            "sparse_gqa": 3,
        }
    )
    assert Counter(case.dataset for case in mi250) == Counter(
        {"deepseek": 6, "vit": 6, "mamba2": 6, "retnet_recur": 4, "mla": 3}
    )
    assert len({case.case_id for case in h100}) == len(h100)
    assert len({case.case_id for case in mi250}) == len(mi250)
    assert all(
        case.require_grad is False
        for case in h100
        if case.attn_type == "causal_softmax_attn" and case.seqlen_q == 1
    )
    assert {case.chunk_size for case in h100 if case.dataset == "mamba2"} == {64}
    assert {case.chunk_size for case in mi250 if case.dataset == "mamba2"} == {32}


def test_dump_bench_result_preserves_schema_and_order(tmp_path):
    fwd_path, bwd_path = dump_bench_result(
        "demo",
        [
            ("BS1\nS128", {"MetaAttention": (1.25, 2.5), "Baseline": (3.0, None)}),
            ("BS8\nS256", {"MetaAttention": (4.0, 5.0)}),
        ],
        tmp_path,
    )
    assert fwd_path.read_text(encoding="utf-8").splitlines() == [
        "Method,BS1 S128,BS8 S256",
        "MetaAttention,1.25,4.0",
        "Baseline,3.0,",
    ]
    assert bwd_path.read_text(encoding="utf-8").splitlines() == [
        "Method,BS1 S128,BS8 S256",
        "MetaAttention,2.5,5.0",
        "Baseline,,",
    ]


def test_load_csv_data_contract(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "Method,BS1 S128,BS8 S256,BS8S1 KV4096\nMetaAttention,1.5,invalid,3.5\n",
        encoding="utf-8",
    )
    providers, data = load_csv_data(path)
    assert providers == ["BS1\nS128", "BS8\nS256", "BS8S1\nKV4096"]
    assert data == [("MetaAttention", [1.5, 0.0, 3.5])]
    providers, data = load_csv_data(path, exclude_kv=True)
    assert providers == ["BS1\nS128", "BS8\nS256"]
    assert data == [("MetaAttention", [1.5, 0.0])]
    assert load_csv_data(tmp_path / "missing.csv") == ([], [])
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert load_csv_data(empty) == ([], [])


@pytest.mark.parametrize(
    ("module", "attn_type", "function_name"),
    [
        (figure11, "causal_softmax_attn", "bench_softmaxattention"),
        (figure11, "sigmoid_attn", "bench_sigmoidattention"),
        (figure11, "gated_retention", "bench_gated_retention"),
        (figure11, "relu_attn", "bench_reluattention"),
        (figure11, "retention_parallel", "bench_retention_parallel"),
        (figure11, "retention_recurrent", "bench_retnet_recurrent"),
        (figure11, "mamba2_ssm", "bench_mamba2_ssm"),
        (figure11, "mla_attn", "bench_mla_decode"),
        (figure11, "sparse_gqa", "bench_sparse_gqa_decode"),
        (figure14, "causal_softmax_attn", "bench_softmaxattention"),
        (figure14, "relu_attn", "bench_reluattention"),
        (figure14, "retention_recurrent", "bench_retnet_recurrent"),
        (figure14, "mamba2_ssm", "bench_mamba2_ssm"),
        (figure14, "mla_attn", "bench_mla_decode"),
    ],
)
def test_bench_attention_dispatch(monkeypatch, module, attn_type, function_name):
    calls = []

    def fake_benchmark(*args, **kwargs):
        calls.append((args, kwargs))
        return {"MetaAttention": (1.0, None)}

    monkeypatch.setattr(module, function_name, fake_benchmark)
    result = module.bench_attention(
        attn_type,
        1,
        2,
        4,
        4,
        8,
        8,
        head_k=1,
        head_v=1,
        timer=lambda callback, **kwargs: 0.0,
    )
    assert result == {"MetaAttention": (1.0, None)}
    assert len(calls) == 1
    assert calls[0][1]["timer"] is not None


def test_bench_attention_unknown_type_is_a_noop(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("benchmark function unexpectedly called")

    monkeypatch.setattr(figure11, "_resolve_timer", fail_if_called)
    assert figure11.bench_attention("unknown", 1, 1, 1, 1, 1, 1) == {}


def test_h100_plot_writes_pdf_and_uses_retention_parallel_backward(
    tmp_path, monkeypatch
):
    result_dir = tmp_path / "h100"
    _write_synthetic_csvs(result_dir, H100_FILES)
    observed = []
    original_loader = plot_fig_h100.plot_figure11.__globals__["load_csv_data"]

    def recording_loader(filename, exclude_kv=False):
        observed.append(Path(filename).name)
        return original_loader(filename, exclude_kv=exclude_kv)

    monkeypatch.setitem(
        plot_fig_h100.plot_figure11.__globals__, "load_csv_data", recording_loader
    )
    output_path = tmp_path / "figure11_h100.pdf"
    plot_fig_h100.plot_figure11(result_dir, output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert "retnet_bwd.csv" in observed
    assert "retnet_recur_bwd.csv" in observed


def test_mi250_plot_writes_pdf(tmp_path):
    result_dir = tmp_path / "mi250"
    _write_synthetic_csvs(result_dir, MI250_FILES)
    output_path = tmp_path / "figure14_mi250.pdf"
    plot_figure14(result_dir, output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_synthetic_benchmark_artifacts_have_metaattention_rows(tmp_path):
    result_dir = tmp_path / "artifacts"
    _write_synthetic_csvs(result_dir, H100_FILES)
    _assert_metaattention_rows(tuple(sorted(result_dir.glob("*.csv"))))


def test_gdn_baseline_reports_complete_staged_forward():
    baseline_path = Path("testing/results/gdn_h20_baseline.json")
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    stages = payload["forward_stage_ms"]
    assert set(stages) == {"gate_cumsum", "kkt", "wy", "state_output"}
    assert all(value > 0 for value in stages.values())
    assert payload["staged_forward_ms"] == pytest.approx(sum(stages.values()))
    assert payload["end_to_end_ms"] > 0
