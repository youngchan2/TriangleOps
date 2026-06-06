import pytest
import torch
from conftest import TOL, ref_attn_pair_bias, requires_cuda, rnd

import triangle_ops


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [128, 256, 512])
def test_matches_fp32_reference(device, dtype, M):
    H, D, Cz = 4, 32, 128
    N = H * D
    X = rnd(M, N, dtype=dtype, device=device, seed=1)
    Z = rnd(M, M, Cz, dtype=dtype, device=device, seed=2)
    WQ = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=3)
    WK = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=4)
    WV = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=5)
    W_ln = torch.ones(Cz, device=device, dtype=dtype)
    B_ln = torch.zeros(Cz, device=device, dtype=dtype)
    W_pz = rnd(Cz, H, dtype=dtype, device=device, sd=Cz**-0.5, seed=6)
    B_pz = torch.zeros(H, device=device, dtype=dtype)
    W_pg = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=7)
    B_pg = torch.zeros(N, device=device, dtype=dtype)
    W_po = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=8)
    B_po = torch.zeros(N, device=device, dtype=dtype)

    out = triangle_ops.attention_pair_bias(
        X,
        WQ,
        WK,
        WV,
        Z,
        W_ln,
        B_ln,
        W_pz,
        B_pz,
        W_pg,
        B_pg,
        W_po,
        B_po,
        H,
        D,
        scale=1.0,
        eps=1e-5,
    )
    ref = ref_attn_pair_bias(
        X,
        WQ,
        WK,
        WV,
        Z,
        W_ln,
        B_ln,
        W_pz,
        B_pz,
        W_pg,
        B_pg,
        W_po,
        B_po,
        H,
        D,
        scale=1.0,
        eps=1e-5,
    )
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs < TOL[dtype], f"M={M} dtype={dtype} max_abs={max_abs:.3e}"


@requires_cuda
def test_precompute_forward_matches_oneshot(device):
    H, D, Cz, M = 4, 32, 128, 256
    N = H * D
    dtype = torch.bfloat16
    X = rnd(M, N, dtype=dtype, device=device, seed=1)
    Z = rnd(M, M, Cz, dtype=dtype, device=device, seed=2)
    WQ, WK, WV = (rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=s) for s in (3, 4, 5))
    W_ln = torch.ones(Cz, device=device, dtype=dtype)
    B_ln = torch.zeros(Cz, device=device, dtype=dtype)
    W_pz = rnd(Cz, H, dtype=dtype, device=device, sd=Cz**-0.5, seed=6)
    B_pz = torch.zeros(H, device=device, dtype=dtype)
    W_pg = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=7)
    B_pg = torch.zeros(N, device=device, dtype=dtype)
    W_po = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=8)
    B_po = torch.zeros(N, device=device, dtype=dtype)

    pre = triangle_ops.attn_pair_bias.precompute(WQ, WK, WV, W_ln, B_ln, W_pz, B_pz, H, D)
    out_amortized = triangle_ops.attn_pair_bias.forward(
        X, Z, pre, scale=1.0, W_proj_g=W_pg, B_proj_g=B_pg, W_proj_o=W_po, B_proj_o=B_po
    )
    out_oneshot = triangle_ops.attention_pair_bias(
        X, WQ, WK, WV, Z, W_ln, B_ln, W_pz, B_pz, W_pg, B_pg, W_po, B_po, H, D, scale=1.0
    )
    assert torch.equal(out_amortized, out_oneshot)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("M", [256, 512])
def test_general_affine_and_bias(device, dtype, M):
    """Trained regime: non-identity LN(z) gamma/beta AND non-zero projection
    biases (B_proj_z folded into the kernel's b_const; B_proj_g/B_proj_o through
    the PyTorch gate/Wo). Verifies absorption + bias terms together."""
    H, D, Cz = 4, 32, 128
    N = H * D
    X = rnd(M, N, dtype=dtype, device=device, seed=1)
    Z = rnd(M, M, Cz, dtype=dtype, device=device, seed=2)
    WQ, WK, WV = (rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=s) for s in (3, 4, 5))
    W_ln = 1.0 + rnd(Cz, dtype=dtype, device=device, sd=0.2, seed=21)
    B_ln = rnd(Cz, dtype=dtype, device=device, sd=0.1, seed=22)
    W_pz = rnd(Cz, H, dtype=dtype, device=device, sd=Cz**-0.5, seed=6)
    B_pz = rnd(H, dtype=dtype, device=device, sd=0.1, seed=23)
    W_pg = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=7)
    B_pg = rnd(N, dtype=dtype, device=device, sd=0.1, seed=24)
    W_po = rnd(N, N, dtype=dtype, device=device, sd=N**-0.5, seed=8)
    B_po = rnd(N, dtype=dtype, device=device, sd=0.1, seed=25)

    out = triangle_ops.attention_pair_bias(
        X, WQ, WK, WV, Z, W_ln, B_ln, W_pz, B_pz, W_pg, B_pg, W_po, B_po, H, D, scale=1.0, eps=1e-5
    )
    ref = ref_attn_pair_bias(
        X, WQ, WK, WV, Z, W_ln, B_ln, W_pz, B_pz, W_pg, B_pg, W_po, B_po, H, D, scale=1.0, eps=1e-5
    )
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs < TOL[dtype], f"M={M} dtype={dtype} max_abs={max_abs:.3e}"
