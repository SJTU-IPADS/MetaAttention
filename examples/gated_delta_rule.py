from __future__ import annotations

import torch

from attn_engine import GDNEngine


def gated_delta_rule(device: str | torch.device = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-length GDN forward/backward with recurrent state on H20."""
    device = torch.device(device)
    batch, query_heads, value_heads, length, dim = 1, 1, 2, 64, 128
    query = torch.randn(batch, query_heads, length, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(batch, value_heads, length, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    gate = (-torch.rand(batch, value_heads, length, device=device)).requires_grad_()
    beta = torch.rand(batch, value_heads, length, device=device, requires_grad=True)
    initial_state = torch.zeros(
        batch, value_heads, dim, dim, device=device, dtype=torch.float32, requires_grad=True
    )

    output, final_state = GDNEngine(device)(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    (output.float().square().mean() + final_state.square().mean()).backward()
    return output, final_state


if __name__ == "__main__":
    result, state = gated_delta_rule()
    print(f"output={tuple(result.shape)} final_state={tuple(state.shape)}")
