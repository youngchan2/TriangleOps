"""Shared benchmark machinery: CUDA-event timing, input/weight generation per op,
and the op/reference runner registry used by sweep.py."""

import math
import statistics
from typing import Callable

import torch

import triangle_ops
from . import references as ref


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def time_ms(fn: Callable[[], object], iters: int = 30, warmup: int = 12) -> float:
    """Median latency in ms via CUDA events; NaN on OOM."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return float("nan")
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) if ts else float("nan")


def _rnd(*shape, dtype, device, sd, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device, dtype=dtype) * sd


# --------------------------------------------------------------------------- #
# per-op: build (op_runner, cueq_runner, torch_runner) for a given size
# --------------------------------------------------------------------------- #
def build_attn_pair_bias(size, dtype, device, *, H=4, D=32, Cz=128, **kw):
    M = size
    N = H * D
    X = _rnd(M, N, dtype=dtype, device=device, sd=0.05, seed=1)
    Z = _rnd(M, M, Cz, dtype=dtype, device=device, sd=0.05, seed=2)
    WQ, WK, WV = (_rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=s) for s in (3, 4, 5))
    W_ln = torch.ones(Cz, device=device, dtype=dtype); B_ln = torch.zeros(Cz, device=device, dtype=dtype)
    W_pz = _rnd(Cz, H, dtype=dtype, device=device, sd=Cz**-0.5, seed=6); B_pz = torch.zeros(H, device=device, dtype=dtype)
    W_pg = _rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=7); B_pg = torch.zeros(N, device=device, dtype=dtype)
    W_po = _rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=8); B_po = torch.zeros(N, device=device, dtype=dtype)
    args = (X, WQ, WK, WV, Z, W_ln, B_ln, W_pz, B_pz, W_pg, B_pg, W_po, B_po, H, D)

    # excl-pc: precompute once, time only forward
    pre = triangle_ops.attn_pair_bias.precompute(WQ, WK, WV, W_ln, B_ln, W_pz, B_pz, H, D)
    fwd = lambda: triangle_ops.attn_pair_bias.forward(
        X, Z, pre, scale=1.0, W_proj_g=W_pg, B_proj_g=B_pg, W_proj_o=W_po, B_proj_o=B_po)
    return {
        "triangle_ops": fwd,
        "triangle_ops_incl": lambda: triangle_ops.attention_pair_bias(*args, scale=1.0),
        "cueq": lambda: ref.cueq_attn_pair_bias(*args, scale=1.0),
        "torch": lambda: ref.torch_attn_pair_bias(*args, scale=1.0),
    }


def build_triangle_attn(size, dtype, device, *, H=4, D=32, Cin=128, **kw):
    N = size
    scale = 1.0 / math.sqrt(D)
    X = _rnd(N, N, Cin, dtype=dtype, device=device, sd=0.05, seed=1)
    W_ln = torch.ones(Cin, device=device, dtype=dtype); B_ln = torch.zeros(Cin, device=device, dtype=dtype)
    WQ, WK, WV = (_rnd(Cin, H * D, dtype=dtype, device=device, sd=Cin**-0.5, seed=s) for s in (3, 4, 5))
    W_pz = _rnd(Cin, H, dtype=dtype, device=device, sd=Cin**-0.5, seed=6); B_pz = torch.zeros(H, device=device, dtype=dtype)
    W_pg = _rnd(Cin, H * D, dtype=dtype, device=device, sd=Cin**-0.5, seed=7)
    W_po = _rnd(Cin, H * D, dtype=dtype, device=device, sd=(H * D)**-0.5, seed=8)
    args = (X, W_ln, B_ln, WQ, WK, WV, W_pz, B_pz, W_pg, None, W_po, None, H, D)

    pre = triangle_ops.triangle_attn.precompute(W_ln, B_ln, WQ, WK, WV, W_pz, B_pz, H, D)
    fwd = lambda: triangle_ops.triangle_attn.forward(
        X, pre, mask=None, scale=scale, W_proj_g=W_pg, B_proj_g=None, W_proj_o=W_po, B_proj_o=None)
    return {
        "triangle_ops": fwd,
        "triangle_ops_incl": lambda: triangle_ops.triangle_attention(*args, scale=scale, mask=None),
        "cueq": lambda: ref.cueq_triangle_attention(*args, scale=scale, mask=None),
        "torch": lambda: ref.torch_triangle_attention(*args, scale=scale, mask=None),
    }


def build_triangle_mul(size, dtype, device, *, D=128, direction="outgoing", **kw):
    L = size
    x = _rnd(1, L, L, D, dtype=dtype, device=device, sd=0.05, seed=1)
    mask = torch.ones(1, L, L, dtype=torch.bool, device=device)
    w = {
        "norm_in_weight": torch.ones(D, device=device, dtype=dtype),
        "norm_in_bias": torch.zeros(D, device=device, dtype=dtype),
        "p_in_weight": _rnd(2 * D, D, dtype=dtype, device=device, sd=D**-0.5, seed=11),
        "g_in_weight": _rnd(2 * D, D, dtype=dtype, device=device, sd=D**-0.5, seed=12),
        "norm_out_weight": torch.ones(D, device=device, dtype=dtype),
        "norm_out_bias": torch.zeros(D, device=device, dtype=dtype),
        "p_out_weight": _rnd(D, D, dtype=dtype, device=device, sd=D**-0.5, seed=13),
        "g_out_weight": _rnd(D, D, dtype=dtype, device=device, sd=D**-0.5, seed=14),
    }
    pre = triangle_ops.triangle_mul.precompute(
        w["norm_in_weight"], w["norm_in_bias"], w["p_in_weight"], w["g_in_weight"],
        w["norm_out_weight"], w["norm_out_bias"], w["p_out_weight"], w["g_out_weight"])
    fwd = lambda: triangle_ops.triangle_mul.forward(x, pre, direction=direction, mask=mask)
    return {
        "triangle_ops": fwd,
        "triangle_ops_incl": lambda: triangle_ops.triangle_multiplicative_update(
            x, direction=direction, mask=mask, **w),
        "cueq": lambda: ref.cueq_triangle_multiplicative_update(x, direction=direction, mask=mask, **w),
        "torch": lambda: ref.torch_triangle_multiplicative_update(x, direction=direction, mask=mask, **w),
    }


BUILDERS = {
    "attn_pair_bias": build_attn_pair_bias,
    "triangle_attn": build_triangle_attn,
    "triangle_mul": build_triangle_mul,
}
