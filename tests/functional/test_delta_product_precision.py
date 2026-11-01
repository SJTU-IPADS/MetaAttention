"""Delta-product inference against an independent ordered residual recurrence."""

import pytest
import torch

from attn_engine import StateTuple, Target
from tests.unit.test_delta_family_examples import _case, _reference

pytestmark = [pytest.mark.functional, pytest.mark.gpu]


@pytest.mark.parametrize("length", [1, 16, 17, 128, 129, 1023, 1024, 1025])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_delta_product_precision_at_chunk_and_window_boundaries(
    gpu_device, length, dtype
):
    torch.manual_seed(42)
    operator, sizes, inputs, initial = _case(
        "delta_product", device=gpu_device, dim=64, dim_value=128, dtype=dtype, steps=3
    )
    inputs = {
        name: tensor.repeat(1, 1, (length + 4) // 5, *([1] * (tensor.ndim - 3)))[
            :, :, :length
        ].contiguous()
        for name, tensor in inputs.items()
    }
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected_output, expected_states = _reference(
            "delta_product",
            {name: tensor.cpu() for name, tensor in inputs.items()},
            {name: tensor.cpu() for name, tensor in initial.items()},
            steps=3,
        )
    finally:
        torch.set_num_threads(previous_threads)
    compiled = operator.compile(role_sizes=sizes, target=Target(), mode="dense")
    state = StateTuple(tuple(initial), tuple(initial.values()))
    state_before = initial["memory"].clone()
    output_tolerance = (
        {"rtol": 0.02, "atol": 0.05}
        if dtype == torch.bfloat16
        else {"rtol": 2e-4, "atol": 2e-5}
    )

    def check(result, output):
        torch.testing.assert_close(
            result.outputs["output"].cpu().float(), output, **output_tolerance
        )
        torch.testing.assert_close(
            result.final_state["memory"].cpu(),
            expected_states["memory"],
            rtol=2e-4,
            atol=2e-5,
        )

    with torch.inference_mode():
        full = compiled(**inputs, initial_state=state, return_final_state=True)
        check(full, expected_output)
        if length == 1025:
            first = compiled(
                **{
                    name: tensor[:, :, :1001].contiguous()
                    for name, tensor in inputs.items()
                },
                initial_state=state,
                return_final_state=True,
            )
            last = compiled(
                **{
                    name: tensor[:, :, 1001:].contiguous()
                    for name, tensor in inputs.items()
                },
                initial_state=first.final_state,
                return_final_state=True,
            )
            check(last, expected_output[:, :, 1001:])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = compiled(
                    **inputs, initial_state=state, return_final_state=True
                )
            graph.replay()
            check(captured, expected_output)
    torch.testing.assert_close(initial["memory"], state_before, rtol=0, atol=0)
