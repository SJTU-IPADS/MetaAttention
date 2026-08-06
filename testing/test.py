from examples.mha import causal_softmax_attention
from examples.mha_v2 import causal_softmax_attention as causal_softmax_attention_v2
from examples.mha_decode import softmax_attention_decode
from examples.mamba2 import mamba2
from examples.gated_retention import gated_retention
from examples.sigmoid_attn import sigmoid_attention
from examples.sparse_gqa_decode import sparse_gqa_decode
from examples.retnet_recurrent import retnet_recurrent
from examples.reluattn import relu_attention
from examples.mla_decode import mla_decode

import torch
import torch.nn.functional as F
from einops import rearrange, einsum, repeat
import math
from typing import Optional
import time

from benchmark.bench_utils import print_debug, assert_close

from termcolor import cprint, colored

def run_test_with_info(func, *args, **kwargs):
    func_name = func.__name__
    
    # blue
    cprint(f"➤ [START] Testing: {func_name}", "cyan", attrs=["bold"])
    
    start_time = time.perf_counter()
    try:
        func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # green
        print(colored(f"✔ [PASS] {func_name}", "green", attrs=["bold"]) + 
              colored(f" ({duration:.4f}s)", "yellow"))
              
    except Exception as e:
        # red
        cprint(f"✘ [FAIL] {func_name} failed!", "red", attrs=["bold"])
        cprint(f"  Error: {str(e)}", "red")
        raise e
    finally:
        print("-" * 60)

def test_attention():
    total_start = time.perf_counter()
    
    cprint("\n" + "="*60, "magenta", attrs=["bold"])
    cprint("      STARTING TEST SUITE", "magenta", attrs=["bold"])
    cprint("="*60 + "\n", "magenta", attrs=["bold"])

    run_test_with_info(test_softmaxattention, 1, 16, 2048, 128, 128) #1 , TODO: fix autotune
    run_test_with_info(test_softmaxattention, 1, 16, 2048, 128, 256) # 1
    run_test_with_info(test_softmaxattention_decode, 8, 16, 1, 4096, 128, 128) # 1
    run_test_with_info(test_mamba2, 1, 1, 2048, 128, 64, HK=1, HV=80) # 1
    run_test_with_info(test_gated_retention, 8, 32, 2048, 256, 256) # 1
    run_test_with_info(test_sigmoid_attention, 1, 32, 2048, 128, 128) # 1
    run_test_with_info(test_sparse_gqa_decode, 8, 32, 8, 2048, 128, 128) # 1
    run_test_with_info(test_retnet_recurrent, 1, 32, 2048, 256, 512) # 1
    run_test_with_info(test_relu_attention, 1, 6, 2048, 64, 64) # 1
    run_test_with_info(test_mla_decode, 8, 128, 2048, 576, 512, HKV=1)
    
    total_end = time.perf_counter()
    total_time = total_end - total_start
    
    cprint("\n" + "="*60, "green", attrs=["bold"])
    cprint("      ALL TESTS PASSED SUCCESSFULLY", "green", attrs=["bold"])
    cprint(f"      Total Time: {total_time:.4f}s", "green", attrs=["bold"])
    cprint("="*60, "green", attrs=["bold"])
    
def test_softmaxattention(B, H, S, D, DV, device="cuda", dtype=torch.float16, require_grad=True, use_v2=False):
    if use_v2:
        attention_module = causal_softmax_attention_v2(B, H, S, D, DV)
    else:
        attention_module = causal_softmax_attention(B, H, S, D, DV, tune=True)
    
    def ref(query, key, value, causal=True, softmax_scale=None):
        dim = query.shape[-1]
        num_head_groups = query.shape[2] // key.shape[2]
        if softmax_scale is None:
            softmax_scale = 1 / dim** 0.5

        query = rearrange(
            query, 'b s (h g) d -> b s g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]
        scores = einsum(query, key,
        'b s g h d, b t h d -> b g h s t')
        if causal:
            seqlenq = query.shape[1]
            seqlenk = key.shape[1]
            mask = torch.tril(
                torch.ones(
                    seqlenq, seqlenk, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(
            scores * softmax_scale, dim=-1)

        out = einsum(attention, value,
                'b g h s t, b t h d -> b g h s d')
        out = rearrange(out, 'b g h s d -> b s (h g) d') 
        return out
    
    # init input
    query = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    
    query1 = query.clone().detach().requires_grad_(require_grad)
    key1 = key.clone().detach().requires_grad_(require_grad)
    value1 = value.clone().detach().requires_grad_(require_grad)
    
    ref_o = ref(query, key, value)
    o = attention_module(query1, key1, value1)
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
    
    if require_grad:
        do = torch.randn_like(o)
        o.backward(do, retain_graph=True)
        ref_o.backward(do, retain_graph=True)
        torch.testing.assert_close(query1.grad, query.grad, rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(key1.grad, key.grad, rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(value1.grad, value.grad, rtol=1e-1, atol=1e-1)
    

def test_softmaxattention_decode(B, H, S, KV, D, DV, device="cuda", dtype=torch.float16):
    attention_module = softmax_attention_decode(B, H, S, KV, D, DV)
    
    def ref(query, key, value, softmax_scale=None):
        dim = query.shape[-1]
        num_head_groups = query.shape[2] // key.shape[2]
        if softmax_scale is None:
            softmax_scale = 1 / dim** 0.5

        query = rearrange(
            query, 'b s (h g) d -> b s g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]
        scores = einsum(query, key,
        'b s g h d, b t h d -> b g h s t')
        attention = F.softmax(
            scores * softmax_scale, dim=-1)

        out = einsum(attention, value,
                'b g h s t, b t h d -> b g h s d')
        out = rearrange(out, 'b g h s d -> b s (h g) d') 
        return out
    
    # init input
    query = torch.randn(B, S, H, D, device=device, dtype=dtype)
    key = torch.randn(B, KV, H, D, device=device, dtype=dtype)
    value = torch.randn(B, KV, H, DV, device=device, dtype=dtype)
    ref_o = ref(query, key, value)
    o = attention_module(query, key, value)
    torch.testing.assert_close(o, ref_o, rtol=1e-2, atol=1e-2)
  
def test_mamba2(B, HQ, S, D, DV, HK=None, HV=None, dtype=torch.bfloat16, require_grad=True):
    attention_module = mamba2(B, HQ, S, D, DV, HK=HK, HV=HV, dtype=dtype)
    
    # init input
    query = torch.randn(B, S, HQ, D, device="cuda", dtype=dtype)
    key = 0.5 * torch.randn(B, S, HK, D, device="cuda", dtype=dtype)
    value = torch.randn(B, S, HV, DV, device="cuda", dtype=dtype)
    if require_grad:
        do = 0.1 * torch.randn(B, S, HV,
                                     DV, dtype=dtype, device="cuda")
    A_mamba = 1.5 * torch.rand(HV, dtype=dtype, device="cuda") - 4.0
    # initialize dt
    accum_dtype = torch.float32
    dt_mamba = 0.7 * torch.rand(B, S, HV, dtype=accum_dtype, device="cuda")
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    dt_min = 0.001
    dt_max = 0.1
    dt = torch.exp(
        torch.rand(HV, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))
    dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)
    
    # ours
    q_ours = query.clone().transpose(1, 2).contiguous()
    k_ours = key.clone().transpose(1, 2).contiguous()
    v_ours = value.clone().transpose(1, 2).contiguous()
    A_ours = A_mamba[None, :].clone().contiguous()
    dt_ours = dt_mamba.clone().transpose(1, 2).contiguous()
    if require_grad:
        do_ours = do.clone().transpose(1, 2).contiguous()
    
    q_ours = q_ours.detach().requires_grad_(require_grad)
    k_ours = k_ours.detach().requires_grad_(require_grad)
    v_ours = v_ours.detach().requires_grad_(require_grad)
    A_ours = A_ours.detach().requires_grad_(require_grad)
    dt_ours = dt_ours.detach().requires_grad_(require_grad)
    
    o = attention_module(
        q_ours, k_ours, v_ours, dt_ours, A_ours, dt_ours.to(dtype)
    )
    if require_grad:
        o.backward(do_ours, retain_graph=True)
    
    # mamba2 ref
    
    def chunk_state_ref(B, x, dt, dA_cumsum):
        """
        Argument:
            B: (batch, seqlen, ngroups, headdim)
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
        Return:
            states: (batch, nchunks, nheads, headdim, dstate)
        """
        # Check constraints.
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[-1]
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        assert x.shape == (batch, seqlen, nheads, headdim)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        ngroups = B.shape[2]
        assert nheads % ngroups == 0
        assert B.shape == (batch, seqlen, ngroups, dstate)
        B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if seqlen < nchunks * chunk_size:
            x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
            B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
        B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
        decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
        return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)

    def state_passing_ref(states, dA_chunk_cumsum, initial_states=None):
        """
        Argument:
            states: (batch, nchunks, nheads, dim)
            dA_chunk_cumsum: (batch, nheads, nchunks)
            initial_states: (batch, nheads, dim)
        Return:
            out: (batch, nchunks, nheads, dim)
            final_states: (batch, nheads, dim)
        """
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, 0])
        states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
        dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
        dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
        nchunks = dA_chunk_cumsum.shape[-1]
        # (batch, nheads, nchunks, nchunks)
        dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
        # (batch, nheads, nchunks, nchunks)
        decay_chunk = torch.exp(dt_chunk_segment_sum)
        causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
        decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
        out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
        return out[:, :-1], out[:, -1]

    def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
        """
        Argument:
            B: (batch, seqlen, ngroups, dstate)
            C: (batch, seqlen, ngroups, dstate)
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
            prev_states: (batch, nchunks, nheads, headdim, dstate)
            D: (nheads, headdim) or (nheads,)
            z: (batch, seqlen, nheads, headdim)
        Return:
            out: (batch, seqlen, nheads, headdim)
        """
        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen == nchunks * chunk_size
        assert C.shape == B.shape
        B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
        C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
        CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                        rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
        # (batch, nheads, nchunks, chunksize, chunksize)
        dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
        decay = torch.exp(dt_segment_sum)
        scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
        scores_decay = scores_decay.masked_fill(~causal_mask, 0)
        out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                        rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
        state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
        out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                                prev_states.to(C.dtype)) * state_decay_out
        out = out + out_prev
        out = rearrange(out, "b c l h p -> b (c l) h p")
        if D is not None:
            if D.dim() == 1:
                D = rearrange(D, "h -> h 1")
            out = out + x * D
        return out if z is None else out * F.silu(z)
    
    @torch.compile
    def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
        """
        Argument:
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, seqlen, nheads)
            A: (nheads)
            B: (batch, seqlen, ngroups, dstate)
            C: (batch, seqlen, ngroups, dstate)
            D: (nheads, headdim) or (nheads,)
            z: (batch, seqlen, nheads, headdim)
            dt_bias: (nheads,)
        Return:
            out: (batch, seqlen, nheads, headdim)
        """
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[-1]
        if seqlen % chunk_size != 0:
            dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
        dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
        dt = dt.float()  # We want high precision for this before cumsum
        if dt_bias is not None:
            dt = dt + rearrange(dt_bias, "h -> h 1 1")
        if dt_softplus:
            dt = F.softplus(dt)
        dA = dt * rearrange(A, "h -> h 1 1")
        dA_cumsum = torch.cumsum(dA, dim=-1)
        # 1. Compute the state for each chunk
        states = chunk_state_ref(B, x, dt, dA_cumsum)
        states_dtype = states.dtype
        if states.dtype not in [torch.float32, torch.float64]:
            states = states.to(torch.float32)
        # 2. Pass the state to all the chunks by weighted cumsum.
        # state_passing_ref is much less numerically stable
        states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                        "... (p n) -> ... p n", n=dstate)
        states = states.to(states_dtype)
        # 3. Compute the output for each chunk
        out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
        return out

    value = value.detach_().requires_grad_(require_grad)
    A_mamba.detach_().requires_grad_(require_grad)
    dt_mamba.detach_().requires_grad_(require_grad)
    key.detach_().requires_grad_(require_grad)
    query.detach_().requires_grad_(require_grad)
    
    query.grad = key.grad = value.grad = None

    out_ref = ssd_chunk_scan_combined_ref(
        value,
        dt_mamba,
        A_mamba,
        key,
        query,
        chunk_size=64).to(dtype)
    if require_grad:
        out_ref.backward(do, retain_graph=True)
        
    assert_close(o.transpose(1, 2), out_ref, rtol=1e-1, atol=1e-1, mismatch_ratio=1e-2)
    if require_grad:
        assert_close(
            q_ours.grad.transpose(1, 2),
            query.grad,
            rtol=1e-1,
            atol=1e-1,
            mismatch_ratio=1e-2
        )
        assert_close(
            k_ours.grad.transpose(1, 2),
            key.grad,
            rtol=1e-1,
            atol=1e-1,
            mismatch_ratio=1e-2
        )
        assert_close(
            v_ours.grad.transpose(1, 2),
            value.grad,
            rtol=1e-1,
            atol=1e-1,
            mismatch_ratio=1e-2
        )
        # pytorch reference has nan, so we skip this check for now
        # torch.testing.assert_close(
        #     A_ours.grad.squeeze(0),
        #     A_mamba.grad,
        #     rtol=1e-1,
        #     atol=1e-1,
        # )
        # torch.testing.assert_close(
        #     dt_ours.grad.transpose(1, 2),
        #     dt_mamba.grad,
        #     rtol=1e-1,
        #     atol=1e-1,
        # )
    
   
def test_gated_retention(B, H, S, D, DV, dtype=torch.bfloat16, require_grad=True):
    
    # prepare input
    accum_dtype = torch.float32
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    g = F.logsigmoid(torch.randn(B, H, S, device="cuda", dtype=accum_dtype)).clamp_min(-5)
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    
    q.detach_().requires_grad_(require_grad)
    k.detach_().requires_grad_(require_grad)
    g.detach_().requires_grad_(require_grad)
    v.detach_().requires_grad_(require_grad)


    q1 = q.clone()
    k1 = k.clone()
    v1 = v.clone()
    g1 = g.clone().to(dtype)
    
    q1.detach_().requires_grad_(require_grad)
    k1.detach_().requires_grad_(require_grad)
    g1.detach_().requires_grad_(require_grad)
    v1.detach_().requires_grad_(require_grad)
    
    # ours
    attention_module = gated_retention(B, H, S, D, DV, dtype=dtype)
    o = attention_module(q, k, v, g)
    if require_grad:
        o.backward(do, retain_graph=True)
    
    
    # from fla.ops.simple_gla import chunk_simple_gla
    # from fla.ops.simple_gla.naive import naive_chunk_simple_gla
    def naive_chunk_simple_gla(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        scale: Optional[float] = None,
    ):
        q, k, v, g = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), g.to(torch.float32)
        if scale is None:
            scale = 1.0 / q.shape[-1] ** 0.5

        T = q.shape[-2]
        BT = chunk_size
        pad_len = (BT - (T % BT)) % BT
        if pad_len > 0:
            # Pad all tensors
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            g = F.pad(g, (0, pad_len))
        decay = g
        B, H, T1, K = q.shape
        V = v.shape[-1]
        q = q * scale
        q, k, v, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, decay.unsqueeze(-1)])
        decay = decay.squeeze(-1).cumsum(-1)
        L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
        S = k.new_zeros(B, H, K, V)
        if initial_state is not None:
            S = initial_state
        o = torch.zeros_like(v)
        for i in range(0, T1 // chunk_size):
            q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i])
            o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
            o[:, :, i] = o_inter + attn @ v_i
            S = S * decay[:, :, i, -1, None, None].exp() + \
                (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_i
        if not output_final_state:
            S = None
        # unpad
        o = rearrange(o, 'b h n c d -> b h (n c) d')[:, :, :T]
        return o, S

    
    o_ref = naive_chunk_simple_gla(q1, k1, v1, g1, chunk_size=64)[0].to(dtype)
    if require_grad:
        o_ref.backward(do, retain_graph=True)
    
    torch.testing.assert_close(o, o_ref, rtol=1e-1, atol=1e-1)
    if require_grad:
        torch.testing.assert_close(
            q.grad,
            q1.grad,
            rtol=1e-1,
            atol=1e-1,
        )
        torch.testing.assert_close(
            k.grad,
            k1.grad,
            rtol=1e-1,
            atol=1e-1,
        )
        torch.testing.assert_close(
            v.grad,
            v1.grad,
            rtol=1e-1,
            atol=1e-1,
        )
        # pytorch reference has nan, so we skip this check for now
        # torch.testing.assert_close(
        #     g.grad,
        #     g1.grad.to(accum_dtype),
        #     rtol=3e-2,
        #     atol=1e-2,
        # )
    
def test_sigmoid_attention(B, H, S, D, DV, device="cuda", dtype=torch.float16, require_grad=True):
    attention_module = sigmoid_attention(B, H, S, D, DV, tune=True)
    
    def ref(query, key, value, sigmoid_bias, causal=True):
        dim = query.shape[-1]
        num_head_groups = query.shape[2] // key.shape[2]

        query = rearrange(
            query, 'b s (h g) d -> b s g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]
        scores = einsum(query, key,
        'b s g h d, b t h d -> b g h s t')
        if causal:
            seqlenq = query.shape[1]
            seqlenk = key.shape[1]
            mask = torch.tril(
                torch.ones(
                    seqlenq, seqlenk, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores += sigmoid_bias
        # attention = torch.sigmoid(scores)
        attention = (torch.tanh(scores*0.5) + 1) * 0.5

        out = einsum(attention, value,
                'b g h s t, b t h d -> b g h s d')
        out = rearrange(out, 'b g h s d -> b s (h g) d') 
        return out
    
    # from flash_sigmoid import flash_attn_func
    # def ref(query, key, value, sigmoid_bias, causal=True):
    #     ref_out = flash_attn_func(
    #         query, key, value, softmax_scale=1.0, causal=causal, sigmoid_bias=sigmoid_bias.to("cpu")
    #     )
    #     return ref_out
    
    accum_dtype = torch.float32
    # init input
    query = torch.randn(B, S, H, D, device=device, dtype=dtype)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype)
    softmax_bias = torch.tensor([1], device=device, dtype=accum_dtype).uniform_(-10., 2.)
    
    
    query.detach_().requires_grad_(require_grad)
    key.detach_().requires_grad_(require_grad)
    value.detach_().requires_grad_(require_grad)
    
    query1 = query.clone().detach().requires_grad_(require_grad)
    key1 = key.clone().detach().requires_grad_(require_grad)
    value1 = value.clone().detach().requires_grad_(require_grad)
    
    ref_o = ref(query, key, value, softmax_bias)
    o = attention_module(query1, key1, value1, softmax_bias)
    
    torch.testing.assert_close(o, ref_o, rtol=1e-1, atol=1e-1)
    
    if require_grad:
        do = 0.1 * torch.randn(B, S, H, DV, device=device, dtype=dtype)
        o.backward(do, retain_graph=True)
        ref_o.backward(do, retain_graph=True)
        torch.testing.assert_close(
            query.grad,
            query1.grad,
            rtol=1e-1,
            atol=1e-1,
        )
        torch.testing.assert_close(
            key.grad,
            key1.grad,
            rtol=1e-1,
            atol=1e-1,
        )
        torch.testing.assert_close(
            value.grad,
            value1.grad,
            rtol=1e-1,
            atol=1e-1,
        )

def test_relu_attention(B, H, S, D, DV, device="cuda", dtype=torch.float16, require_grad=True):
    attention_module = relu_attention(B, H, S, D, DV, dtype=dtype, tune=True)
    
    def ref(query, key, value):
        qk = torch.einsum('bqhd,bkhd->bhqk', query, key)
        qk = qk / (D ** 0.5)
        qk = F.relu(qk)
        o = torch.einsum('bhqk,bkhd->bqhd', qk, value)
        return o
    
    accum_dtype = torch.float32
    # init input
    query = torch.randn(B, S, H, D, device=device, dtype=dtype)
    key = 0.5 * torch.randn(B, S, H, D, device=device, dtype=dtype)
    value = 0.5 * torch.randn(B, S, H, DV, device=device, dtype=dtype)
    
    query = query.detach_().requires_grad_(require_grad)
    key = key.detach_().requires_grad_(require_grad)
    value = value.detach_().requires_grad_(require_grad)
    
    ref_o = ref(query, key, value)
    
    query1 = query.clone().detach().requires_grad_(require_grad)
    key1 = key.clone().detach().requires_grad_(require_grad)
    value1 = value.clone().detach().requires_grad_(require_grad)
    o = attention_module(query1, key1, value1)
    
    assert_close(o, ref_o, rtol=1e-1, atol=1e-1, mismatch_ratio=1e-3)
    
    if require_grad:
        do = 0.1 * torch.randn(B, S, H, DV, device=device, dtype=dtype)
        o.backward(do, retain_graph=True)
        ref_o.backward(do, retain_graph=True)
        assert_close(
            query.grad,
            query1.grad,
            rtol=1e-1,
            atol=1e-1,
            mismatch_ratio=1e-3
        )
        assert_close(
            key.grad,
            key1.grad,
            rtol=1e-1,
            atol=1e-1,
            mismatch_ratio=1e-3
        )
        assert_close(
            value.grad,
            value1.grad,
            rtol=1e-1,
            atol=1e-1,
            mismatch_ratio=1e-3
        )

def test_sparse_gqa_decode(B, H, G, S, D, DV, device="cuda", dtype=torch.float16):
    attention_module = sparse_gqa_decode(B, H, G, S, D, DV, dtype=dtype)
    
    def ref_program_torch(query, key, value, block_mask, cache_seqlens, max_cache_seqlen, num_blocks,
                      block_size):
        query = query.squeeze(1)  # [batch_size, heads, dim]
        batch, heads, dim = query.shape
        heads_kv = key.shape[2]

        num_head_groups = query.shape[1] // key.shape[2]
        scale = dim**0.5
        key = rearrange(key, 'b n h d -> b h n d')  # [batch_size, heads_kv, seqlen_kv, dim]
        value = rearrange(value, 'b n h d -> b h n d')  # [batch_size, heads_kv, seqlen_kv, dim]

        query = rearrange(
            query, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, heads_kv, dim]

        scores = einsum(
            query, key,
            'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

        sparse_mask = torch.zeros_like(scores)
        # Assign mask values
        for b in range(batch):
            for h in range(heads_kv):
                for idx in range(num_blocks):
                    if block_mask[b, h, idx]:
                        sparse_mask[b, :, h, idx * block_size:(idx + 1) * block_size] = 1

        scores = scores.masked_fill(sparse_mask == 0, float('-inf'))

        range_len = torch.arange(scores.shape[-1], device='cuda').unsqueeze(0)
        cache_seqlens_expanded = cache_seqlens.unsqueeze(1)
        pad_mask = range_len >= cache_seqlens_expanded
        pad_mask = pad_mask[:, None, None, :]
        scores = scores.masked_fill(pad_mask, float('-inf'))
        attention = F.softmax(
            scores / scale, dim=-1)  # [batch_size, num_head_groups, heads_kv, seqlen_kv]

        out = einsum(attention, value,
                    'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, heads_kv, dim]
        out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
        return out

    block_size = 32
    sparse_ratio = 0.5
    # init input
    query = torch.randn(B, 1, H, D, device=device, dtype=dtype)
    key = torch.randn(B, S, G, D, device=device, dtype=dtype)
    value = torch.randn(B, S, G, DV, device=device, dtype=dtype)
    # cache_seqlens = torch.randint(1, S, (B,), dtype=torch.int32, device=device)
    # random_index = torch.randint(0, B, (1,), device='cuda').item()  # Select a random index
    # cache_seqlens[random_index] = S  # Assign cache_seqlen to ensure at least one occurrence
    cache_seqlens = torch.full((B,), S, dtype=torch.int32, device='cuda')
    
    def generate_block_mask(batch, heads_kv, max_cache_seqlen, sparse_ratio, cache_seqlens):
        num_blocks = (max_cache_seqlen + block_size - 1) // block_size

        valid_num_blocks = torch.ceil(cache_seqlens * (1 - sparse_ratio) / block_size).int()
        # print("valid_num_blocks: ", valid_num_blocks)
        max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
        # print("max_valid_num_blocks: ", max_valid_num_blocks)
        # Initialize block_mask with false (for padding blocks)
        block_mask = torch.zeros((batch, heads_kv, num_blocks), dtype=torch.bool, device='cuda')

        # Assign valid indices while ensuring no duplicates within each batch-group
        for b in range(batch):
            max_valid_block = max_valid_num_blocks[b].item()  # Max valid blocks for this batch
            valid_num_block = valid_num_blocks[b].item()  # Valid blocks for this batch
            if valid_num_block > 0:  # Ensure there's at least one valid block
                for h in range(heads_kv):
                    perm = torch.randperm(max_valid_block, device='cuda')[:valid_num_block]
                    block_mask[b, h, perm] = True
                    
        return block_mask
    block_mask = generate_block_mask(B, G, S, sparse_ratio, cache_seqlens)
    
    ref_o = ref_program_torch(query, key, value, block_mask, cache_seqlens, S,
                              (S + block_size - 1) // block_size, block_size)
    o = attention_module(query, key, value, block_mask=block_mask, cache_seqlens=cache_seqlens)
    
    torch.testing.assert_close(o, ref_o, rtol=1e-1, atol=1e-1)
    
def test_retnet_recurrent(B, H, S, D, DV, dtype=torch.bfloat16, require_grad=True):
    attention_module = retnet_recurrent(B, H, S, D, DV, dtype=dtype)
    
    def ref(q, k, v):
        orig_type = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        _, n_heads, seq_len, d_head = q.shape
        s = (1 - q.new_tensor(2., dtype=torch.float).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
        n = q.new_tensor(range(seq_len), dtype=torch.float)
        n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
        s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
        o = torch.einsum('bhqk,bhkd->bhqd', s, v)
        return o.to(orig_type)
    
    # init input
    accum_dtype = torch.float32
    query = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    key = 0.1 * torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    g = torch.tensor(range(0, H), dtype=accum_dtype)
    g = 1 - torch.exp2(-5 - g)
    g = g[None, :, None].expand(B, H, S).cuda().detach().contiguous()
    value = torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    do = 0.1 * torch.randn(B, H, S, DV, device="cuda", dtype=dtype)
    
    query.detach_().requires_grad_(require_grad)
    key.detach_().requires_grad_(require_grad)
    g.detach_().requires_grad_(require_grad)
    value.detach_().requires_grad_(require_grad)
    
    q1 = query.clone()
    k1 = key.clone()
    v1 = value.clone()
    g1 = g.clone()
    q1.detach_().requires_grad_(require_grad)
    k1.detach_().requires_grad_(require_grad)
    g1.detach_().requires_grad_(False)
    v1.detach_().requires_grad_(require_grad)
    
    # ours
    o = attention_module(query, key, value, g)
    if require_grad:
        o.backward(do, retain_graph=True)
        
    # ref
    o_ref = ref(q1, k1, v1)
    if require_grad:
        o_ref.backward(do, retain_graph=True)
    torch.testing.assert_close(o, o_ref, rtol=1e-1, atol=1e-1)
    if require_grad:
        torch.testing.assert_close(
            query.grad,
            q1.grad,
            rtol=1e-1,
            atol=1e-1,
        )
        torch.testing.assert_close(
            key.grad,
            k1.grad,
            rtol=1e-1,
            atol=1e-1,
        )
        torch.testing.assert_close(
            value.grad,
            v1.grad,
            rtol=1e-1,
            atol=1e-1,
        )

def test_mla_decode(B, HQ, SKV, D, DV, HKV=1, dtype=torch.float16):
    
    attention_module = mla_decode(B, HQ, SKV, D, DV, HK=HKV, HV=HKV, dtype=dtype, tune=True)
    
    def ref(q, q_pe, kv, k_pe):
        q = q.squeeze(1)  # [batch_size, heads, dim]
        q_pe = q_pe.squeeze(1)  # [batch_size, heads, pe_dim]
        dim = q.shape[-1]
        pe_dim = q_pe.shape[-1]
        num_head_groups = q.shape[1] // kv.shape[2]
        scale = (dim + pe_dim)**0.5
        q = rearrange(
            q, 'b (h g) d -> b g h d', g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

        q_pe = rearrange(
            q_pe, 'b (h g) d -> b g h d',
            g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

        kv = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

        k_pe = rearrange(k_pe, 'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, pe_dim]

        query = torch.concat([q, q_pe], dim=-1)
        key = torch.concat([kv, k_pe], dim=-1)

        scores = einsum(
            query, key,
            'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(
            scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(attention, kv,
                    'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
        out = out.unsqueeze(1)
        return out
        
    
    q = torch.randn(B, 1, HQ, DV, device="cuda", dtype=dtype)
    q_pe = torch.randn(B, 1, HQ, D-DV, device="cuda", dtype=dtype)
    kv = torch.randn(B, SKV, HKV, DV, device="cuda", dtype=dtype)
    k_pe = torch.randn(B, SKV, HKV, D-DV, device="cuda", dtype=dtype)
    
    
    o = attention_module(q, q_pe, kv, k_pe)
    o_ref = ref(q, q_pe, kv, k_pe)
    
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    
    
    
    
    
if __name__ == "__main__":
    test_attention()
