from __future__ import annotations


import pytest

from testing.benchmark_stateful_plans import (
    CASES,
    MATRIX_BATCHES,
    MATRIX_HEADS,
    MATRIX_RECORD_FIELDS,
    MATRIX_SEQLENS,
    expand_matrix,
    matrix_record_schema,
)


pytestmark = pytest.mark.unit


def test_matrix_expansion_is_deterministic_and_complete():
    selected = CASES[:2]
    workloads = expand_matrix(selected)
    assert len(workloads) == len(selected) * len(MATRIX_BATCHES) * len(
        MATRIX_HEADS
    ) * len(MATRIX_SEQLENS)
    assert [
        (case.name, case.batch, case.heads, seqlen) for case, seqlen in workloads
    ] == [
        (case.name, batch, heads, seqlen)
        for case in selected
        for batch in MATRIX_BATCHES
        for heads in MATRIX_HEADS
        for seqlen in MATRIX_SEQLENS
    ]
    assert workloads == expand_matrix(selected)


def test_matrix_expansion_preserves_case_dimensions_for_non_square_state():
    gsa = next(case for case in CASES if case.name == "gsa")
    expanded = expand_matrix((gsa,), batches=(2,), heads=(16,), seqlens=(65,))
    case, seqlen = expanded[0]
    assert (
        case.batch,
        case.heads,
        case.dim,
        case.dim_slot,
        case.dim_value,
        seqlen,
    ) == (
        2,
        16,
        gsa.dim,
        gsa.dim_slot,
        gsa.dim_value,
        65,
    )


def test_matrix_expansion_rejects_empty_or_nonpositive_axes():
    with pytest.raises(ValueError, match="must not be empty"):
        expand_matrix(CASES[:1], batches=())
    with pytest.raises(ValueError, match="must be positive"):
        expand_matrix(CASES[:1], seqlens=(0,))


def test_matrix_record_schema_is_explicit_and_json_safe():
    record = {field: field for field in MATRIX_RECORD_FIELDS}
    record["unrelated"] = object()
    assert matrix_record_schema(record) == {
        field: field for field in MATRIX_RECORD_FIELDS
    }
    assert "meta_correctness" in MATRIX_RECORD_FIELDS
