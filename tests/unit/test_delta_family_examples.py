"""Example semantics against independent token recurrences, through public calls."""

from __future__ import annotations

import importlib

import pytest
import torch

from attn_engine import FeatureRole, HeadRole, StateTuple, Target


pytestmark = [pytest.mark.unit, pytest.mark.examples]


@pytest.fixture(autouse=True)
def _allow_cpu_stateful_operator(monkeypatch, request):
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setenv("STATEFUL_OPERATOR_TEST_CPU", "1")


FAMILIES = (
    "delta_rule",
    "kda",
    "gdn2",
    "comba",
    "iplr",
    "dplr",
    "oja",
    "delta_product",
    "momentum",
    "abc",
)
FACTORIES = {
    "delta_rule": ("delta_rule", "delta_rule_attention"),
    "kda": ("kda", "kda"),
    "gdn2": ("gdn2", "gdn2"),
    "comba": ("comba", "comba"),
    "iplr": ("generalized_delta_rule", "iplr_delta_rule"),
    "dplr": ("generalized_delta_rule", "dplr_delta_rule"),
    "oja": ("gated_oja_rule", "gated_oja_rule"),
    "delta_product": ("gated_delta_product", "gated_delta_product"),
    "momentum": ("momentum_delta_rule", "momentum_delta_rule"),
    "abc": ("abc_attention", "abc_attention"),
}


def _case(kind, *, device="cpu", dim=3, dim_value=5, dtype=torch.float32, steps=2):
    module_name, factory_name = FACTORIES[kind]
    module = importlib.import_module(f"examples.{module_name}")
    factory = getattr(module, factory_name)
    args = (1, 2, 5, dim, 4, dim_value) if kind == "abc" else (1, 2, 5, dim, dim_value)
    kwargs = {"num_householder": steps} if kind == "delta_product" else {}
    operator = factory(*args, dtype=dtype, **kwargs)
    role_sizes = {
        module.KEY: dim,
        module.VALUE: dim_value,
        module.QUERY_HEADS: 1,
        module.KEY_HEADS: 1,
        module.VALUE_HEADS: 2,
        module.STATE_HEADS: 2,
    }
    if kind == "abc":
        role_sizes[module.SLOT] = 4
    generator = torch.Generator().manual_seed(73)
    inputs = {}
    for spec in operator.algorithm.inputs:
        shape = tuple(
            role_sizes[role]
            if isinstance(role, (FeatureRole, HeadRole))
            else 1
            if role.name == "batch"
            else 5
            for role in spec.roles
        )
        value = 0.2 * torch.randn(shape, generator=generator)
        if spec.name in ("gate", "log_alpha", "log_mu"):
            value = -0.3 * torch.rand(shape, generator=generator)
        elif spec.name.startswith("beta") and kind not in ("iplr", "dplr"):
            value = 0.1 + 0.4 * torch.rand(shape, generator=generator)
        elif spec.name in ("erase", "write"):
            value = torch.rand(shape, generator=generator)
        inputs[spec.name] = value.to(device=device, dtype=spec.dtype)
    states = {}
    for spec in operator.algorithm.states:
        shape = (1, 2, *(role_sizes[role] for role in spec.feature_roles))
        value = 0.1 * torch.randn(shape, generator=generator)
        if spec.name == "normalizer":
            value = torch.ones(shape)
        states[spec.name] = value.to(device)
    return operator, role_sizes, inputs, states


def _reference(kind, inputs, initial, *, steps=2):
    """Direct residual updates, not the expanded propagations in the examples."""
    # Contiguous head groups deliberately differ between query/key and State.
    heads = next(iter(initial.values())).shape[1]
    values = {
        name: value.float().repeat_interleave(heads // value.shape[1], dim=1)
        for name, value in inputs.items()
    }
    states = {name: value.float().clone() for name, value in initial.items()}
    output = []
    dim = values["query"].shape[-1]
    for token in range(values["query"].shape[2]):
        x = {name: value[:, :, token] for name, value in values.items()}
        q = x["query"] * dim**-0.5
        if kind == "abc":
            weight = x["slot_logits"].exp()
            states["key_slot"] = states["key_slot"] + x["key"].unsqueeze(
                -1
            ) * weight.unsqueeze(-2)
            states["slot_value"] = states["slot_value"] + weight.unsqueeze(-1) * x[
                "value"
            ].unsqueeze(-2)
            states["normalizer"] = states["normalizer"] + weight
            keys = states["key_slot"] / states["normalizer"].unsqueeze(-2)
            slot_values = states["slot_value"] / states["normalizer"].unsqueeze(-1)
            probabilities = torch.einsum("bhk,bhkm->bhm", q, keys).softmax(-1)
            output.append(torch.einsum("bhm,bhmv->bhv", probabilities, slot_values))
            continue
        state = states["memory"]
        if kind == "momentum":
            prediction = torch.einsum("bhk,bhkv->bhv", x["prediction_key"], state)
            momentum = x["log_mu"].exp()[..., None, None] * states["momentum"]
            momentum = momentum + x["eta"][..., None, None] * x["key"].unsqueeze(-1) * (
                prediction - x["value"]
            ).unsqueeze(-2)
            states["momentum"] = momentum
            state = (
                x["log_alpha"].exp()[..., None, None] * state
                - x["beta"][..., None, None] * momentum
            )
        elif kind in ("iplr", "dplr"):
            correction = torch.einsum("bhk,bhkv->bhv", x["alpha"], state)
            if kind == "dplr":
                state = state * x["gate"].exp().unsqueeze(-1)
            state = state + x["beta"].unsqueeze(-1) * correction.unsqueeze(-2)
            state = state + x["key"].unsqueeze(-1) * x["value"].unsqueeze(-2)
        elif kind == "oja":
            state = state * x["gate"].exp().unsqueeze(-2)
            residual = x["key"] - torch.einsum("bhkv,bhv->bhk", state, x["value"])
            state = state + (x["beta"].unsqueeze(-1) * residual).unsqueeze(-1) * x[
                "value"
            ].unsqueeze(-2)
        elif kind == "delta_product":
            state = state * x["gate"].exp()[..., None, None]
            for step in range(steps):
                key, value, beta = (
                    x[f"{name}_{step}"] for name in ("key", "value", "beta")
                )
                residual = value - torch.einsum("bhk,bhkv->bhv", key, state)
                state = state + key.unsqueeze(-1) * (
                    beta.unsqueeze(-1) * residual
                ).unsqueeze(-2)
        else:
            if kind in ("kda", "gdn2"):
                state = state * x["gate"].exp().unsqueeze(-1)
            elif kind == "comba":
                state = state * x["gate"].exp()[..., None, None]
            prediction_key = x["prediction_key"] if kind == "comba" else x["key"]
            if kind == "gdn2":
                prediction_key = prediction_key * x["erase"]
            prediction = torch.einsum("bhk,bhkv->bhv", prediction_key, state)
            residual = (
                x["write"] * x["value"] - prediction
                if kind == "gdn2"
                else x["beta"].unsqueeze(-1) * (x["value"] - prediction)
            )
            state = state + x["key"].unsqueeze(-1) * residual.unsqueeze(-2)
        states["memory"] = state
        output.append(torch.einsum("bhk,bhkv->bhv", q, state))
    return torch.stack(output, dim=2), states


def _invoke(
    operator, roles, inputs, initial, *, device="cpu", backward=False, mode="dense"
):
    compiled = operator.compile(
        role_sizes=roles,
        target=Target(device_type=device),
        mode=mode,
        backward=backward,
    )
    return compiled(
        **inputs,
        initial_state=StateTuple(tuple(initial), tuple(initial.values())),
        return_final_state=True,
    )


@pytest.mark.parametrize("kind", FAMILIES)
def test_examples_match_recurrence_and_continue_state(kind):
    operator, roles, inputs, initial = _case(kind)
    actual = _invoke(operator, roles, inputs, initial)
    expected_output, expected_states = _reference(kind, inputs, initial)
    torch.testing.assert_close(
        actual.outputs["output"], expected_output, rtol=2e-5, atol=1e-6
    )
    for name, expected in expected_states.items():
        torch.testing.assert_close(
            actual.final_state[name], expected, rtol=2e-5, atol=1e-6
        )
    prefix = _invoke(
        operator,
        roles,
        {name: value[:, :, :2].contiguous() for name, value in inputs.items()},
        initial,
    )
    suffix = _invoke(
        operator,
        roles,
        {name: value[:, :, 2:].contiguous() for name, value in inputs.items()},
        prefix.final_state.as_dict(),
    )
    torch.testing.assert_close(
        suffix.outputs["output"],
        actual.outputs["output"][:, :, 2:],
        rtol=2e-5,
        atol=1e-6,
    )
    for name in expected_states:
        torch.testing.assert_close(
            suffix.final_state[name], actual.final_state[name], rtol=2e-5, atol=1e-6
        )


@pytest.mark.parametrize("kind", FAMILIES)
def test_examples_start_from_zero_when_initial_state_is_omitted(kind):
    operator, roles, inputs, initial = _case(kind)
    compiled = operator.compile(
        role_sizes=roles, target=Target(device_type="cpu"), mode="dense"
    )
    actual = compiled(**inputs, return_final_state=True)
    zeros = {name: torch.zeros_like(value) for name, value in initial.items()}
    output, states = _reference(kind, inputs, zeros)
    torch.testing.assert_close(actual.outputs["output"], output, rtol=2e-5, atol=1e-6)
    for name, expected in states.items():
        torch.testing.assert_close(
            actual.final_state[name], expected, rtol=2e-5, atol=1e-6
        )


@pytest.mark.parametrize("kind", FAMILIES)
def test_example_gradients_match_recurrence(kind):
    operator, roles, inputs, initial = _case(kind)
    inputs = {name: value.requires_grad_() for name, value in inputs.items()}
    initial = {name: value.requires_grad_() for name, value in initial.items()}
    actual = _invoke(operator, roles, inputs, initial, backward=True)
    expected_output, expected_states = _reference(kind, inputs, initial)
    weights = torch.linspace(-1, 1, expected_output.numel()).reshape_as(expected_output)
    actual_loss = (actual.outputs["output"] * weights).sum()
    expected_loss = (expected_output * weights).sum()
    for name, expected in expected_states.items():
        actual_loss = actual_loss + actual.final_state[name].square().sum() * 0.1
        expected_loss = expected_loss + expected.square().sum() * 0.1
    leaves = (*inputs.values(), *initial.values())
    actual_gradients = torch.autograd.grad(actual_loss, leaves)
    expected_gradients = torch.autograd.grad(expected_loss, leaves)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-4, atol=2e-6
        )


@pytest.mark.parametrize("steps", (1, 3, 12))
def test_delta_product_preserves_all_intermediate_writes(steps):
    operator, roles, inputs, initial = _case("delta_product", steps=steps)
    actual = _invoke(operator, roles, inputs, initial)
    output, states = _reference("delta_product", inputs, initial, steps=steps)
    torch.testing.assert_close(actual.outputs["output"], output, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(
        actual.final_state["memory"], states["memory"], rtol=2e-5, atol=1e-6
    )


@pytest.mark.parametrize("steps", (0, -1, 1.5, True))
def test_delta_product_rejects_invalid_step_counts(steps):
    from examples.gated_delta_product import gated_delta_product

    with pytest.raises(ValueError, match="positive integer"):
        gated_delta_product(1, 2, 5, 3, 5, num_householder=steps)


def test_legacy_delta_rule_entry_point_uses_true_delta_semantics():
    from examples.linear_attention import delta_rule_attention

    operator, roles, inputs, initial = _case("delta_rule")
    legacy = delta_rule_attention(1, 2, 5, 3, 5, kv_heads=1, dtype=torch.float32)
    actual = _invoke(legacy, roles, inputs, initial)
    expected = _invoke(operator, roles, inputs, initial)
    torch.testing.assert_close(actual.outputs["output"], expected.outputs["output"])
    torch.testing.assert_close(
        actual.final_state["memory"], expected.final_state["memory"]
    )


@pytest.mark.gpu
@pytest.mark.parametrize("kind", FAMILIES)
def test_examples_execute_on_gpu(kind, gpu_device):
    operator, roles, inputs, initial = _case(
        kind, device=gpu_device, dim=64, dim_value=128, dtype=torch.bfloat16
    )
    actual = _invoke(operator, roles, inputs, initial, device="cuda")
    output, states = _reference(kind, inputs, initial)
    torch.testing.assert_close(
        actual.outputs["output"].float(), output, rtol=0.06, atol=0.01
    )
    for name, expected in states.items():
        torch.testing.assert_close(
            actual.final_state[name], expected, rtol=0.06, atol=0.02
        )


@pytest.mark.gpu
@pytest.mark.parametrize("kind", FAMILIES)
def test_example_gradients_execute_on_gpu(kind, gpu_device):
    operator, roles, inputs, initial = _case(
        kind, device=gpu_device, dim=64, dim_value=128, dtype=torch.bfloat16
    )
    inputs = {name: value.requires_grad_() for name, value in inputs.items()}
    initial = {name: value.requires_grad_() for name, value in initial.items()}
    actual = _invoke(operator, roles, inputs, initial, device="cuda", backward=True)
    output, states = _reference(kind, inputs, initial)
    weights = torch.linspace(-1, 1, output.numel(), device=gpu_device).reshape_as(
        output
    )
    actual_loss = (actual.outputs["output"].float() * weights).sum()
    expected_loss = (output * weights).sum()
    for name, expected in states.items():
        actual_loss = actual_loss + actual.final_state[name].square().sum() * 0.1
        expected_loss = expected_loss + expected.square().sum() * 0.1
    leaves = (*inputs.values(), *initial.values())
    actual_gradients = torch.autograd.grad(actual_loss, leaves)
    expected_gradients = torch.autograd.grad(expected_loss, leaves)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0.08, atol=0.02
        )


def test_momentum_decayed_prediction_matches_recurrence():
    from examples.momentum_delta_rule import momentum_delta_rule

    _, roles, inputs, initial = _case("momentum")
    operator = momentum_delta_rule(
        1, 2, 5, 3, 5, dtype=torch.float32, use_p_times_alpha=True
    )
    actual = _invoke(operator, roles, inputs, initial)
    reference_inputs = dict(inputs)
    # The variant explicitly promotes p to FP32 before scaling.
    reference_inputs["prediction_key"] = inputs["prediction_key"] * inputs[
        "log_alpha"
    ].exp().unsqueeze(-1)
    expected, final = _reference("momentum", reference_inputs, initial)
    torch.testing.assert_close(actual.outputs["output"], expected, rtol=2e-5, atol=2e-6)
    for name, value in final.items():
        torch.testing.assert_close(
            actual.final_state[name], value, rtol=2e-5, atol=2e-6
        )


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ("momentum", "delta_product"))
def test_ordered_examples_packed_empty_sequence_gradients(kind, gpu_device):
    operator, roles, dense_inputs, initial = _case(
        kind,
        device=gpu_device,
        dim=64,
        dim_value=128,
        dtype=torch.bfloat16,
    )
    inputs = {
        name: value.squeeze(0).transpose(0, 1).contiguous().requires_grad_()
        for name, value in dense_inputs.items()
    }
    states = {
        name: value.repeat(3, 1, 1, 1).requires_grad_()
        for name, value in initial.items()
    }
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device=gpu_device)
    compiled = operator.compile(
        role_sizes=roles, target=Target(), mode="packed", backward=True
    )
    actual = compiled(
        **inputs,
        sequence_offsets=offsets,
        initial_state=StateTuple(tuple(states), tuple(states.values())),
        return_final_state=True,
    )
    expected_output, final_parts = [], {name: [] for name in states}
    for index, (start, end) in enumerate(((0, 2), (2, 2), (2, 5))):
        if start == end:
            for name in states:
                final_parts[name].append(states[name][index : index + 1])
            continue
        output, final = _reference(
            kind,
            {
                name: value[start:end].transpose(0, 1).unsqueeze(0)
                for name, value in inputs.items()
            },
            {name: value[index : index + 1] for name, value in states.items()},
        )
        expected_output.append(output.squeeze(0).transpose(0, 1))
        for name in states:
            final_parts[name].append(final[name])
    expected_output = torch.cat(expected_output)
    expected_states = {name: torch.cat(parts) for name, parts in final_parts.items()}
    torch.testing.assert_close(
        actual.outputs["output"].float(), expected_output, rtol=0.06, atol=0.01
    )
    for name in states:
        torch.testing.assert_close(
            actual.final_state[name], expected_states[name], rtol=0.04, atol=0.005
        )
    leaves = (*inputs.values(), *states.values())
    actual_loss = actual.outputs["output"].float().square().sum() + sum(
        x.square().sum() for x in actual.final_state
    )
    expected_loss = expected_output.square().sum() + sum(
        x.square().sum() for x in expected_states.values()
    )
    for observed, expected in zip(
        torch.autograd.grad(actual_loss, leaves),
        torch.autograd.grad(expected_loss, leaves),
    ):
        torch.testing.assert_close(observed, expected, rtol=0.08, atol=0.02)


@pytest.mark.gpu
@pytest.mark.parametrize("loss_kind", ("output", "final", "both"))
def test_momentum_longer_dense_continuation_and_gradients(loss_kind, gpu_device):
    operator, roles, inputs, initial = _case(
        "momentum",
        device=gpu_device,
        dim=64,
        dim_value=128,
        dtype=torch.bfloat16,
    )
    inputs = {
        name: value.repeat(2, 1, 4, *([1] * (value.ndim - 3)))[:, :, :17]
        .contiguous()
        .requires_grad_()
        for name, value in inputs.items()
    }
    initial = {
        name: value.repeat(2, 1, 1, 1).requires_grad_()
        for name, value in initial.items()
    }
    compiled = operator.compile(
        role_sizes=roles, target=Target(), mode="dense", backward=True
    )
    actual = compiled(
        **inputs,
        initial_state=StateTuple(tuple(initial), tuple(initial.values())),
        return_final_state=True,
    )
    output, final = _reference("momentum", inputs, initial)
    torch.testing.assert_close(
        actual.outputs["output"].float(), output, rtol=0.06, atol=0.01
    )
    for name, value in final.items():
        torch.testing.assert_close(
            actual.final_state[name], value, rtol=0.02, atol=0.002
        )
    prefix = compiled(
        **{name: value[:, :, :7].contiguous() for name, value in inputs.items()},
        initial_state=StateTuple(tuple(initial), tuple(initial.values())),
        return_final_state=True,
    )
    suffix = compiled(
        **{name: value[:, :, 7:].contiguous() for name, value in inputs.items()},
        initial_state=prefix.final_state,
        return_final_state=True,
    )
    torch.testing.assert_close(
        suffix.outputs["output"],
        actual.outputs["output"][:, :, 7:],
        rtol=0.06,
        atol=0.01,
    )
    for name in final:
        torch.testing.assert_close(
            suffix.final_state[name], actual.final_state[name], rtol=0.02, atol=0.002
        )
    actual_loss = (
        actual.outputs["output"].float().square().sum() if loss_kind != "final" else 0
    )
    expected_loss = output.square().sum() if loss_kind != "final" else 0
    if loss_kind != "output":
        actual_loss = actual_loss + sum(
            value.square().sum() for value in actual.final_state
        )
        expected_loss = expected_loss + sum(
            value.square().sum() for value in final.values()
        )
    leaves = (*inputs.values(), *initial.values())
    actual_grad = torch.autograd.grad(actual_loss, leaves, allow_unused=True)
    expected_grad = torch.autograd.grad(expected_loss, leaves, allow_unused=True)
    for actual_value, expected_value in zip(actual_grad, expected_grad):
        if expected_value is None:
            assert actual_value is None
        else:
            torch.testing.assert_close(
                actual_value, expected_value, rtol=0.08, atol=0.02
            )
