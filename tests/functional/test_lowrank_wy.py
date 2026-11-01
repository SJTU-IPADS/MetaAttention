"""Consumer-visible ordered rank-one outputs and State continuation."""

from __future__ import annotations

import pytest
import torch

from attn_engine.stateful.lowering.lowrank.lowrank_kernels import execute

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


def _case(device, length, dtype, *, vector_scale=False):
    batch, heads, rows, columns = 2, 2, 32, 48
    generator = torch.Generator(device=device).manual_seed(2119)

    def random(size, scale=1.0):
        return (
            (
                torch.randn(
                    batch, length, heads, size, device=device, generator=generator
                )
                * scale
            )
            .to(dtype)
            .reshape(batch * length, heads, size)
        )

    left = [random(rows, rows**-0.5), random(rows, rows**-0.5)]
    right = [random(rows, rows**-0.5), random(rows, rows**-0.5)]
    coefficients = [random(1, 0.08).float(), random(1, 0.08).float()]
    scale = torch.full((batch, length, heads, 1), 0.98, device=device)
    # An inverse/log-ratio implementation cannot recover after these gates.
    if length > 5:
        scale[:, 5] = 0.0
    if length > 17:
        scale[:, 17] = -0.3
    scale = scale.reshape(batch * length, heads, 1)
    tail_scale = torch.full_like(scale, 0.97)
    if vector_scale:
        scale = scale.expand(-1, -1, rows).clone()
        scale[..., ::2] *= 0.9
    return {
        "operation_codes": (
            "rank-one",
            "row-scale" if vector_scale else "scale",
            "rank-one",
            "scale",
        ),
        "left_factors": [
            left[0],
            scale if vector_scale else torch.ones_like(scale),
            left[1],
            torch.ones_like(tail_scale),
        ],
        "right_factors": [
            right[0],
            torch.ones_like(scale),
            right[1],
            torch.ones_like(tail_scale),
        ],
        "coefficients": [coefficients[0], scale, coefficients[1], tail_scale],
        "injection_left": [random(rows, 0.1), random(rows, 0.1)],
        "injection_right": [random(columns, 0.2), random(columns, 0.2)],
        "readouts": [random(rows, 0.2), random(rows, 0.2)],
        "initial_state": torch.randn(
            batch, heads, rows, columns, device=device, generator=generator
        )
        * 0.1,
        "output_dtypes": (dtype, dtype),
    }


def _oracle(arguments, precision):
    """Independent token recurrence: no chunk summaries or triangular solve."""
    state = arguments["initial_state"].to(device="cpu", dtype=precision)
    batch, heads, rows, columns = state.shape
    length = arguments["readouts"][0].shape[0] // batch

    def at(tensor, token):
        return tensor.cpu().to(precision).reshape(batch, length, heads, -1)[:, token]

    outputs = [[] for _ in arguments["readouts"]]
    for token in range(length):
        for index, code in enumerate(arguments["operation_codes"]):
            if code == "scale":
                state = state * at(arguments["coefficients"][index], token).unsqueeze(
                    -1
                )
            elif code == "row-scale":
                state = state * at(arguments["left_factors"][index], token).unsqueeze(
                    -1
                )
            else:
                left = at(arguments["left_factors"][index], token)
                right = at(arguments["right_factors"][index], token)
                coefficient = at(arguments["coefficients"][index], token)
                response = (state * right.unsqueeze(-1)).sum(-2)
                state = state + (coefficient * left).unsqueeze(-1) * response.unsqueeze(
                    -2
                )
        for key, value in zip(
            arguments["injection_left"], arguments["injection_right"], strict=True
        ):
            state = state + at(key, token).unsqueeze(-1) * at(value, token).unsqueeze(
                -2
            )
        for output, query in zip(outputs, arguments["readouts"], strict=True):
            output.append((state * at(query, token).unsqueeze(-1)).sum(-2))
    return tuple(torch.stack(output, 2) for output in outputs), state


def _slice(arguments, start, stop):
    result = dict(arguments)
    batch = arguments["initial_state"].shape[0]
    length = arguments["readouts"][0].shape[0] // batch
    for name in (
        "left_factors",
        "right_factors",
        "coefficients",
        "injection_left",
        "injection_right",
        "readouts",
    ):
        result[name] = [
            tensor.reshape(batch, length, *tensor.shape[1:])[:, start:stop]
            .contiguous()
            .flatten(0, 1)
            for tensor in arguments[name]
        ]
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 17, 67, 129, 257])
def test_wy_ordered_two_rank_updates_multiple_terms_and_continuation(
    gpu_device, dtype, length
):
    arguments = _case(gpu_device, length, dtype)
    # Verify BF16 input arithmetic independently of output quantization.
    arguments["output_dtypes"] = (torch.float32, torch.float32)
    with torch.no_grad():
        full_precision, full_precision_final = execute(**arguments)
    for precision in (torch.float64, torch.float32):
        expected, expected_final = _oracle(arguments, precision)
        for actual, reference in zip(full_precision, expected, strict=True):
            torch.testing.assert_close(
                actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
            )
        torch.testing.assert_close(
            full_precision_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
        )
    arguments["output_dtypes"] = (dtype, dtype)
    with torch.no_grad():
        output, final = execute(**arguments)
    for actual, reference in zip(output, expected, strict=True):
        if dtype == torch.bfloat16:
            # Tiny FP32 errors can cross a BF16 midpoint. Bound error against
            # the unrounded oracle, not one arbitrarily selected rounded side.
            rounded = reference.to(dtype)
            above = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
            below = torch.nextafter(rounded, torch.full_like(rounded, float("-inf")))
            ulp = torch.maximum(
                above.float() - rounded.float(), rounded.float() - below.float()
            )
            bound = ulp.double() * 0.5 + 2e-6 + reference.abs() * 2e-5
            assert torch.all((actual.cpu().double() - reference).abs() <= bound)
        else:
            torch.testing.assert_close(
                actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
            )
    torch.testing.assert_close(
        final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )
    assert final.dtype == torch.float32
    if length > 1:
        split = length // 2
        prefix_args = _slice(arguments, 0, split)
        suffix_args = _slice(arguments, split, length)
        with torch.no_grad():
            prefix, prefix_final = execute(**prefix_args)
            suffix_args["initial_state"] = prefix_final
            suffix, suffix_final = execute(**suffix_args)
        for actual, first, last in zip(output, prefix, suffix, strict=True):
            if dtype == torch.bfloat16:
                # Each partition may round to either side of a midpoint.
                joined = torch.cat((first, last), 2)
                above = torch.nextafter(actual, torch.full_like(actual, float("inf")))
                below = torch.nextafter(actual, torch.full_like(actual, float("-inf")))
                ulp = torch.maximum(
                    above.float() - actual.float(), actual.float() - below.float()
                )
                assert torch.all((joined.float() - actual.float()).abs() <= ulp + 2e-6)
            else:
                torch.testing.assert_close(
                    torch.cat((first, last), 2), actual, rtol=2e-4, atol=2e-4
                )
        torch.testing.assert_close(suffix_final, final, rtol=2e-4, atol=2e-4)
    if length == 257 and dtype == torch.float32:
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            captured, captured_final = execute(**arguments)
        graph.replay()
        for actual, reference in zip(captured, expected, strict=True):
            torch.testing.assert_close(
                actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
            )
        torch.testing.assert_close(
            captured_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
        )


@pytest.mark.parametrize("length", [1, 19, 65, 130])
def test_noncommuting_row_scale_output_continuation_and_graph(gpu_device, length):
    arguments = _case(gpu_device, length, torch.float32, vector_scale=True)
    initial_before = arguments["initial_state"].clone()
    with torch.no_grad():
        output, final = execute(**arguments)
    expected, expected_final = _oracle(arguments, torch.float64)
    for actual, reference in zip(output, expected, strict=True):
        torch.testing.assert_close(
            actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        arguments["initial_state"], initial_before, rtol=0, atol=0
    )
    if length > 1:
        split = min(63, length - 1)
        prefix_args = _slice(arguments, 0, split)
        suffix_args = _slice(arguments, split, length)
        with torch.no_grad():
            prefix, carried = execute(**prefix_args)
            suffix_args["initial_state"] = carried
            suffix, split_final = execute(**suffix_args)
        for first, last, reference in zip(prefix, suffix, expected, strict=True):
            torch.testing.assert_close(
                torch.cat((first, last), 2).cpu(),
                reference.float(),
                rtol=2e-4,
                atol=2e-4,
            )
        torch.testing.assert_close(
            split_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
        )
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        captured, captured_final = execute(**arguments)
    graph.replay()
    for actual, reference in zip(captured, expected, strict=True):
        torch.testing.assert_close(
            actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        captured_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )


@pytest.mark.parametrize("length", [19, 257])
def test_reversing_two_arbitrary_rank_updates_changes_outputs(gpu_device, length):
    arguments = _case(gpu_device, length, torch.float32)
    reversed_arguments = dict(arguments)
    for name in ("left_factors", "right_factors", "coefficients"):
        values = list(arguments[name])
        values[0], values[2] = values[2], values[0]
        reversed_arguments[name] = values
    with torch.no_grad():
        forward, final = execute(**arguments)
        reverse, reverse_final = execute(**reversed_arguments)
    expected, expected_final = _oracle(reversed_arguments, torch.float64)
    for actual, reference in zip(reverse, expected, strict=True):
        torch.testing.assert_close(
            actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        reverse_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )
    assert not torch.allclose(final, reverse_final, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(forward[0], reverse[0], rtol=1e-6, atol=1e-7)


def test_single_rank_injection_and_readout_with_partial_chunk(gpu_device):
    arguments = _case(gpu_device, 67, torch.float32)
    indices = (1, 0)
    arguments["operation_codes"] = ("scale", "rank-one")
    for name in ("left_factors", "right_factors", "coefficients"):
        arguments[name] = [arguments[name][i] for i in indices]
    for name in ("injection_left", "injection_right", "readouts"):
        arguments[name] = arguments[name][:1]
    arguments["output_dtypes"] = (torch.float32,)
    with torch.no_grad():
        output, final = execute(**arguments)
    expected, expected_final = _oracle(arguments, torch.float64)
    torch.testing.assert_close(
        output[0].cpu(), expected[0].float(), rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )


@pytest.mark.parametrize("length", [35, 257])
def test_rank_only_transition_without_injections(gpu_device, length):
    arguments = _case(gpu_device, length, torch.float32)
    arguments["operation_codes"] = ("rank-one",)
    for name in ("left_factors", "right_factors", "coefficients"):
        arguments[name] = arguments[name][:1]
    arguments["injection_left"] = []
    arguments["injection_right"] = []
    initial = arguments["initial_state"].clone()
    with torch.no_grad():
        output, final = execute(**arguments)
    expected, expected_final = _oracle(arguments, torch.float64)
    for actual, reference in zip(output, expected, strict=True):
        torch.testing.assert_close(
            actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(arguments["initial_state"], initial, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [67, 257])
def test_wy_strided_factors_broadcast_heads_and_scalar_gates(gpu_device, dtype, length):
    arguments = _case(gpu_device, length, dtype)
    arguments["output_dtypes"] = (torch.float32, torch.float32)
    for name in (
        "left_factors",
        "right_factors",
        "injection_left",
        "injection_right",
        "readouts",
    ):
        parts = []
        for tensor in arguments[name]:
            storage = torch.empty(
                *tensor.shape[:-1],
                2 * tensor.shape[-1] + 1,
                device=gpu_device,
                dtype=tensor.dtype,
            )
            view = storage[..., 1::2]
            view.copy_(tensor)
            if name == "right_factors":
                view = view[:, :1].expand_as(view)
            parts.append(view)
        arguments[name] = parts
    arguments["coefficients"][0] = arguments["coefficients"][0][:, :1].expand_as(
        arguments["coefficients"][0]
    )
    constant = torch.tensor(0.97, device=gpu_device)
    arguments["coefficients"][-1] = constant
    oracle_arguments = dict(arguments)
    oracle_arguments["coefficients"] = [
        *arguments["coefficients"][:-1],
        constant.expand_as(arguments["coefficients"][1]),
    ]
    expected, expected_final = _oracle(oracle_arguments, torch.float64)
    initial = arguments["initial_state"].clone()
    with torch.no_grad():
        output, final = execute(**arguments)
    for actual, reference in zip(output, expected, strict=True):
        torch.testing.assert_close(
            actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(arguments["initial_state"], initial, rtol=0, atol=0)
    prefix_args = _slice(oracle_arguments, 0, 33)
    suffix_args = _slice(oracle_arguments, 33, length)
    with torch.no_grad():
        prefix, prefix_final = execute(**prefix_args)
        suffix_args["initial_state"] = prefix_final
        suffix, suffix_final = execute(**suffix_args)
    for first, last, reference in zip(prefix, suffix, expected, strict=True):
        torch.testing.assert_close(
            torch.cat((first, last), 2).cpu(),
            reference.float(),
            rtol=2e-4,
            atol=2e-4,
        )
    torch.testing.assert_close(
        suffix_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )


@pytest.mark.parametrize("length", [67, 257])
def test_wy_three_event_padding_with_partial_feature_tiles(gpu_device, length):
    arguments = _case(gpu_device, length, torch.float32)
    # Two rank events plus one injection require 48 real events in a 64 tile.
    arguments["injection_left"] = arguments["injection_left"][:1]
    arguments["injection_right"] = arguments["injection_right"][:1]
    for name in ("left_factors", "right_factors", "injection_left", "readouts"):
        arguments[name] = [tensor[..., :31] for tensor in arguments[name]]
    arguments["initial_state"] = arguments["initial_state"][..., :31, :]
    with torch.no_grad():
        output, final = execute(**arguments)
    expected, expected_final = _oracle(arguments, torch.float64)
    for actual, reference in zip(output, expected, strict=True):
        torch.testing.assert_close(
            actual.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )


@pytest.mark.parametrize("length", [1, 17, 65, 257])
def test_wy_symmetric_rank_one_reuses_left_factor_without_changing_results(
    gpu_device, length
):
    arguments = _case(gpu_device, length, torch.float32)
    arguments["right_factors"] = [
        tensor + 1000.0 for tensor in arguments["left_factors"]
    ]
    expected_arguments = dict(arguments)
    expected_arguments["right_factors"] = list(arguments["left_factors"])
    with torch.no_grad():
        actual, actual_final = execute(**arguments, symmetric_rank_one=True)
        expected, expected_final = _oracle(expected_arguments, torch.float64)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        actual_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )
    if length == 1:
        return
    with torch.no_grad():
        split = max(1, length // 2)
        prefix = _slice(arguments, 0, split)
        suffix = _slice(arguments, split, length)
        prefix_output, prefix_final = execute(**prefix, symmetric_rank_one=True)
        suffix["initial_state"] = prefix_final
        suffix_output, suffix_final = execute(**suffix, symmetric_rank_one=True)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result.cpu(), reference.float(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        actual_final.cpu(), expected_final.float(), rtol=2e-4, atol=2e-4
    )
    for result, first, second in zip(actual, prefix_output, suffix_output, strict=True):
        torch.testing.assert_close(
            torch.cat((first, second), dim=2).cpu(), result.cpu(), rtol=2e-4, atol=2e-4
        )
    torch.testing.assert_close(
        suffix_final.cpu(), actual_final.cpu(), rtol=2e-4, atol=2e-4
    )
