import math
import pytest
import torch

import triangle_ops
from conftest import rnd, ref_triangle_attn, TOL, requires_cuda


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N", [128, 130, 256])      # 130 exercises tail/non-pow2
@pytest.mark.parametrize("masked", [False, True])
def test_matches_fp32_reference(device, dtype, N, masked):
    H, D, Cin = 4, 32, 128
    scale = 1.0 / math.sqrt(D)
    X = rnd(N, N, Cin, dtype=dtype, device=device, seed=1)
    W_ln = torch.ones(Cin, device=device, dtype=dtype); B_ln = torch.zeros(Cin, device=device, dtype=dtype)
    WQ, WK, WV = (rnd(Cin, H * D, dtype=dtype, device=device, sd=Cin**-0.5, seed=s) for s in (3, 4, 5))
    W_pz = rnd(Cin, H, dtype=dtype, device=device, sd=Cin**-0.5, seed=6); B_pz = torch.zeros(H, device=device, dtype=dtype)
    W_pg = rnd(Cin, H * D, dtype=dtype, device=device, sd=Cin**-0.5, seed=7)
    W_po = rnd(Cin, H * D, dtype=dtype, device=device, sd=(H * D)**-0.5, seed=8)

    mask = None
    if masked:
        gmask = torch.Generator(device=device).manual_seed(N)
        mask = torch.rand(N, N, device=device, generator=gmask) > 0.2

    out = triangle_ops.triangle_attention(
        X, W_ln, B_ln, WQ, WK, WV, W_pz, B_pz, W_pg, None, W_po, None,
        H, D, scale=scale, eps=1e-5, mask=mask)
    ref = ref_triangle_attn(
        X, W_ln, B_ln, WQ, WK, WV, W_pz, B_pz, W_pg, None, W_po, None,
        H, D, scale=scale, eps=1e-5, mask=mask)
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs < TOL[dtype], f"N={N} dtype={dtype} masked={masked} max_abs={max_abs:.3e}"
