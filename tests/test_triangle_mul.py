import pytest
import torch

import triangle_ops
from conftest import rnd, ref_triangle_mul, TOL, requires_cuda


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


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("L", [64, 130, 256])
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("masked", [False, True])
def test_matches_fp32_reference(device, dtype, L, direction, masked):
    D = 128
    x = rnd(1, L, L, D, dtype=dtype, device=device, seed=1)
    if masked:
        gm = torch.Generator(device=device).manual_seed(L)
        mask = torch.rand(1, L, L, device=device, generator=gm) > 0.2
    else:
        mask = torch.ones(1, L, L, dtype=torch.bool, device=device)
    w = _weights(D, dtype, device)

    out = triangle_ops.triangle_multiplicative_update(
        x, direction=direction, mask=mask, eps=1e-5, **w)
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
        w["norm_in_weight"], w["norm_in_bias"], w["p_in_weight"], w["g_in_weight"],
        w["norm_out_weight"], w["norm_out_bias"], w["p_out_weight"], w["g_out_weight"])
    out_amortized = triangle_ops.triangle_mul.forward(x, pre, direction="outgoing", mask=mask)
    out_oneshot = triangle_ops.triangle_multiplicative_update(x, direction="outgoing", mask=mask, **w)
    assert torch.equal(out_amortized, out_oneshot)
