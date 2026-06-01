"""Shared test fixtures + pure-torch fp32 references (no cuequivariance dependency)."""

import math
import pytest
import torch
import torch.nn.functional as F

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def rnd(*shape, dtype, device, sd=0.05, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device, dtype=dtype) * sd


# Per-dtype absolute tolerance vs fp32 reference (kernels measured well inside these).
TOL = {torch.bfloat16: 5e-2, torch.float16: 1e-2}


# --------------------------------------------------------------------------- #
# fp32 references (ground truth)
# --------------------------------------------------------------------------- #
def ref_attn_pair_bias(X, WQ, WK, WV, Z, W_ln, B_ln, W_proj_z, B_proj_z,
                       W_proj_g, B_proj_g, W_proj_o, B_proj_o, H, D, scale, eps):
    M, N = X.shape
    Cz = Z.shape[-1]
    Xf = X.float()
    q = (Xf @ WQ.float()).reshape(M, H, D).permute(1, 0, 2)
    k = (Xf @ WK.float()).reshape(M, H, D).permute(1, 0, 2)
    v = (Xf @ WV.float()).reshape(M, H, D).permute(1, 0, 2)
    zln = F.layer_norm(Z.float(), (Cz,), W_ln.float(), B_ln.float(), eps=eps)
    bp = (zln @ W_proj_z.float() + B_proj_z.float()).permute(2, 0, 1)
    scores = torch.matmul(q * scale, k.transpose(-1, -2)) + bp
    o = torch.matmul(torch.softmax(scores, -1), v).permute(1, 0, 2).reshape(M, N)
    bg = None if B_proj_g is None else B_proj_g.float()
    bo = None if B_proj_o is None else B_proj_o.float()
    g = torch.sigmoid(F.linear(Xf, W_proj_g.float(), bg))
    return F.linear(g * o, W_proj_o.float(), bo)


def ref_triangle_attn(X, W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z,
                      W_proj_g, B_proj_g, W_proj_o, B_proj_o, H, D, scale, eps, mask):
    N = X.shape[0]
    Cin = X.shape[-1]
    xln = F.layer_norm(X.float(), (Cin,), W_ln.float(), B_ln.float(), eps=eps)
    q = (xln @ WQ.float()).view(N, N, H, D).permute(0, 2, 1, 3)
    k = (xln @ WK.float()).view(N, N, H, D).permute(0, 2, 1, 3)
    v = (xln @ WV.float()).view(N, N, H, D).permute(0, 2, 1, 3)
    bp = (xln @ W_proj_z.float() + B_proj_z.float()).permute(2, 0, 1).unsqueeze(0)
    scores = torch.matmul(q * scale, k.transpose(-1, -2)) + bp
    if mask is not None:
        scores = scores + torch.where(mask.bool()[:, None, None, :], 0.0, -1e9)
    o = torch.matmul(torch.softmax(scores, -1), v).permute(0, 2, 1, 3).reshape(N, N, H * D)
    bg = None if B_proj_g is None else B_proj_g.float()
    bo = None if B_proj_o is None else B_proj_o.float()
    g = torch.sigmoid(F.linear(X.float(), W_proj_g.float(), bg)).view(N, N, H, D)
    o = (o.view(N, N, H, D) * g).reshape(N, N, H * D)
    return F.linear(o, W_proj_o.float(), bo)


def ref_triangle_mul(x, mask, direction, eps, **w):
    C = x.shape[-1]
    xln = F.layer_norm(x.float(), (C,), w["norm_in_weight"].float(), w["norm_in_bias"].float(), eps=eps)
    x_in = xln
    g = F.linear(xln, w["p_in_weight"].float()) * F.linear(xln, w["g_in_weight"].float()).sigmoid()
    if mask is not None:
        g = g * mask.unsqueeze(-1).float()
    a, b = torch.chunk(g, 2, dim=-1)
    if direction == "outgoing":
        y = torch.einsum("bikd,bjkd->bijd", a, b)
    else:
        y = torch.einsum("bkid,bkjd->bijd", a, b)
    yln = F.layer_norm(y, y.shape[-1:], w["norm_out_weight"].float(), w["norm_out_bias"].float(), eps=eps)
    return F.linear(yln, w["p_out_weight"].float()) * F.linear(x_in, w["g_out_weight"].float()).sigmoid()
