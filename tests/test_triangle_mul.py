import pytest
import torch
from conftest import TOL, ref_triangle_mul, requires_cuda, requires_cueq, rnd

import triangle_ops


def _weights(D, dtype, device):
    return {
        "norm_in_weight": torch.ones(D, device=device, dtype=dtype),
        "norm_in_bias": torch.zeros(D, device=device, dtype=dtype),
        "p_in_weight": rnd(2 * D, D, dtype=dtype, device=device, sd=D**-0.5, seed=11),
        "g_in_weight": rnd(2 * D, D, dtype=dtype, device=device, sd=D**-0.5, seed=12),
        "norm_out_weight": torch.ones(D, device=device, dtype=dtype),
        "norm_out_bias": torch.zeros(D, device=device, dtype=dtype),
        "p_out_weight": rnd(D, D, dtype=dtype, device=device, sd=D**-0.5, seed=13),
        "g_out_weight": rnd(D, D, dtype=dtype, device=device, sd=D**-0.5, seed=14),
    }


@requires_cueq
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("L", [64, 130, 256])
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("masked", [False, True])
def test_matches_cuequiv(device, dtype, L, direction, masked):
    D = 128
    x = rnd(1, L, L, D, dtype=dtype, device=device, seed=1)
    if masked:
        gm = torch.Generator(device=device).manual_seed(L)
        mask = torch.rand(1, L, L, device=device, generator=gm) > 0.2
    else:
        mask = torch.ones(1, L, L, dtype=torch.bool, device=device)
    w = _weights(D, dtype, device)

    out = triangle_ops.triangle_multiplicative_update(x, direction=direction, mask=mask, eps=1e-5, **w)
    ref = ref_triangle_mul(x, mask, direction, 1e-5, **w)
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs < TOL[dtype], f"L={L} {direction} dtype={dtype} masked={masked} max_abs={max_abs:.3e}"


@requires_cuda
def test_precompute_forward_matches_oneshot(device):
    D, L = 128, 256
    dtype = torch.bfloat16
    x = rnd(1, L, L, D, dtype=dtype, device=device, seed=1)
    mask = torch.ones(1, L, L, dtype=torch.bool, device=device)
    w = _weights(D, dtype, device)
    pre = triangle_ops.triangle_mul.precompute(
        w["norm_in_weight"],
        w["norm_in_bias"],
        w["p_in_weight"],
        w["g_in_weight"],
        w["norm_out_weight"],
        w["norm_out_bias"],
        w["p_out_weight"],
        w["g_out_weight"],
    )
    out_amortized = triangle_ops.triangle_mul.forward(x, pre, direction="outgoing", mask=mask)
    out_oneshot = triangle_ops.triangle_multiplicative_update(x, direction="outgoing", mask=mask, **w)
    assert torch.equal(out_amortized, out_oneshot)


@requires_cueq
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("L", [128, 256])
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_general_affine(device, dtype, L, direction):
    """Trained-LN regime: non-identity LayerNorm γ (≠1) and β (≠0) for BOTH
    norm_in and norm_out — exercises the absorption path with real-ish values
    (γ ~ 1+N(0,0.2), β ~ N(0,0.1)) rather than the identity affine."""
    D = 128
    x = rnd(1, L, L, D, dtype=dtype, device=device, seed=1)
    mask = torch.ones(1, L, L, dtype=torch.bool, device=device)
    w = {
        "norm_in_weight": 1.0 + rnd(D, dtype=dtype, device=device, sd=0.2, seed=21),
        "norm_in_bias": rnd(D, dtype=dtype, device=device, sd=0.1, seed=22),
        "p_in_weight": rnd(2 * D, D, dtype=dtype, device=device, sd=D**-0.5, seed=11),
        "g_in_weight": rnd(2 * D, D, dtype=dtype, device=device, sd=D**-0.5, seed=12),
        "norm_out_weight": 1.0 + rnd(D, dtype=dtype, device=device, sd=0.2, seed=23),
        "norm_out_bias": rnd(D, dtype=dtype, device=device, sd=0.1, seed=24),
        "p_out_weight": rnd(D, D, dtype=dtype, device=device, sd=D**-0.5, seed=13),
        "g_out_weight": rnd(D, D, dtype=dtype, device=device, sd=D**-0.5, seed=14),
    }
    out = triangle_ops.triangle_multiplicative_update(x, direction=direction, mask=mask, eps=1e-5, **w)
    ref = ref_triangle_mul(x, mask, direction, 1e-5, **w)
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs < TOL[dtype], f"general-affine L={L} {direction} dtype={dtype} max_abs={max_abs:.3e}"
