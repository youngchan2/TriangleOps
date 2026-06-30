"""Comparison baselines for the benchmarks: cuequivariance (optimized kernels)
and a pure-torch bf16 path.  Kept OUT of the `triangle_ops` library so the
library has no cuequivariance dependency.

Each reference mirrors the signature of the matching `triangle_ops` one-shot API
so the sweep can call op and reference with identical arguments.

NOTE on cuequiv fallback thresholds (eager mode): it falls back to plain torch
below a sequence-length threshold (100), which means triangle_mul L≤100 and
attention_pair_bias M≤100 (pair size M²≤10000) are NOT the optimized kernel.
All benchmark points at/above 128 use the optimized cuequiv kernels.
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# cuequivariance
# --------------------------------------------------------------------------- #
def cueq_attn_pair_bias(
    X,
    WQ,
    WK,
    WV,
    Z,
    W_ln,
    B_ln,
    W_proj_z,
    B_proj_z,
    W_proj_g,
    B_proj_g,
    W_proj_o,
    B_proj_o,
    H,
    D,
    scale=1.0,
    eps=1e-5,
):
    from cuequivariance_torch.primitives.triangle import attention_pair_bias as cueq_apb

    M, N = X.shape
    q = (X @ WQ).reshape(M, H, D).permute(1, 0, 2).contiguous()
    k = (X @ WK).reshape(M, H, D).permute(1, 0, 2).contiguous()
    v = (X @ WV).reshape(M, H, D).permute(1, 0, 2).contiguous()
    mask = torch.ones(1, M, device=X.device, dtype=torch.bool)
    out, _ = cueq_apb(
        s=X.unsqueeze(0),
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        z=Z.unsqueeze(0),
        mask=mask,
        num_heads=H,
        w_proj_z=W_proj_z.t().contiguous(),
        w_proj_g=W_proj_g,
        w_proj_o=W_proj_o,
        w_ln_z=W_ln,
        b_ln_z=B_ln,
        b_proj_z=B_proj_z,
        b_proj_g=B_proj_g,
        b_proj_o=B_proj_o,
        eps=eps,
        attn_scale=scale,
        return_z_proj=False,
    )
    return out[0].to(X.dtype)


def cueq_triangle_attention(
    X,
    W_ln,
    B_ln,
    WQ,
    WK,
    WV,
    W_proj_z,
    B_proj_z,
    W_proj_g,
    B_proj_g,
    W_proj_o,
    B_proj_o,
    H,
    D,
    scale=1.0,
    eps=1e-5,
    mask=None,
):
    from cuequivariance_torch.primitives.triangle import triangle_attention as cueq_tri

    N, _, Cin = X.shape
    xln = F.layer_norm(X, (Cin,), W_ln, B_ln, eps=eps)
    q = (xln @ WQ).view(N, N, H, D).permute(0, 2, 1, 3).contiguous()
    k = (xln @ WK).view(N, N, H, D).permute(0, 2, 1, 3).contiguous()
    v = (xln @ WV).view(N, N, H, D).permute(0, 2, 1, 3).contiguous()
    bp = (xln @ W_proj_z + B_proj_z).permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)
    # mask is (N, N) pair-level (per-row key mask); cuequiv broadcasts it over heads
    # & queries via (B=1, N_row, 1, 1, N_key) — additive -inf inside the kernel.
    m = None if mask is None else mask.bool().view(1, N, 1, 1, N)
    out = cueq_tri(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        bp.unsqueeze(0),
        mask=m,
        scale=scale,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.squeeze(0) if out.dim() == 5 else out
    o = out.permute(0, 2, 1, 3).reshape(N, N, H * D).to(X.dtype)
    # K-Fold's MultiHeadAttention gates on q_x = LN(x), not raw x (cuequiv's
    # triangle_attention kernel is core-only; the gate lives in K-Fold's torch wrapper).
    g = torch.sigmoid(F.linear(xln.to(X.dtype), W_proj_g, B_proj_g)).view(N, N, H, D)
    o = (o.view(N, N, H, D) * g).reshape(N, N, H * D)
    return F.linear(o, W_proj_o, B_proj_o)


def cueq_triangle_multiplicative_update(x, *, direction, mask=None, eps=1e-5, **w):
    from cuequivariance_torch.primitives.triangle import (
        triangle_multiplicative_update as cueq_tmu,
    )

    return cueq_tmu(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=w["norm_in_weight"],
        norm_in_bias=w["norm_in_bias"],
        p_in_weight=w["p_in_weight"],
        g_in_weight=w["g_in_weight"],
        norm_out_weight=w["norm_out_weight"],
        norm_out_bias=w["norm_out_bias"],
        p_out_weight=w["p_out_weight"],
        g_out_weight=w["g_out_weight"],
        eps=eps,
    )


# --------------------------------------------------------------------------- #
# pure torch (bf16) — same-dtype path, for a "no fused kernel" baseline
# --------------------------------------------------------------------------- #
def torch_attn_pair_bias(
    X,
    WQ,
    WK,
    WV,
    Z,
    W_ln,
    B_ln,
    W_proj_z,
    B_proj_z,
    W_proj_g,
    B_proj_g,
    W_proj_o,
    B_proj_o,
    H,
    D,
    scale=1.0,
    eps=1e-5,
):
    M, N = X.shape
    Cz = Z.shape[-1]
    q = (X @ WQ).reshape(M, H, D).permute(1, 0, 2)
    k = (X @ WK).reshape(M, H, D).permute(1, 0, 2)
    v = (X @ WV).reshape(M, H, D).permute(1, 0, 2)
    bp = (F.layer_norm(Z, (Cz,), W_ln, B_ln, eps=eps) @ W_proj_z + B_proj_z).permute(2, 0, 1)
    scores = torch.matmul(q * scale, k.transpose(-1, -2)) + bp
    o = torch.matmul(torch.softmax(scores, -1), v).permute(1, 0, 2).reshape(M, N)
    g = torch.sigmoid(F.linear(X, W_proj_g, B_proj_g))
    return F.linear(g * o, W_proj_o, B_proj_o)


def torch_triangle_attention(
    X,
    W_ln,
    B_ln,
    WQ,
    WK,
    WV,
    W_proj_z,
    B_proj_z,
    W_proj_g,
    B_proj_g,
    W_proj_o,
    B_proj_o,
    H,
    D,
    scale=1.0,
    eps=1e-5,
    mask=None,
):
    N, _, Cin = X.shape
    xln = F.layer_norm(X, (Cin,), W_ln, B_ln, eps=eps)
    q = (xln @ WQ).view(N, N, H, D).permute(0, 2, 1, 3)
    k = (xln @ WK).view(N, N, H, D).permute(0, 2, 1, 3)
    v = (xln @ WV).view(N, N, H, D).permute(0, 2, 1, 3)
    bp = (xln @ W_proj_z + B_proj_z).permute(2, 0, 1).unsqueeze(0)
    scores = torch.matmul(q * scale, k.transpose(-1, -2)) + bp
    if mask is not None:
        scores = scores + torch.where(mask.bool()[:, None, None, :], 0.0, -1e9)
    o = torch.matmul(torch.softmax(scores, -1), v).permute(0, 2, 1, 3).reshape(N, N, H * D)
    g = torch.sigmoid(F.linear(xln, W_proj_g, B_proj_g)).view(N, N, H, D)  # gate on LN(x)
    o = (o.view(N, N, H, D) * g).reshape(N, N, H * D)
    return F.linear(o, W_proj_o, B_proj_o)


def torch_triangle_multiplicative_update(x, *, direction, mask=None, eps=1e-5, **w):
    C = x.shape[-1]
    xln = F.layer_norm(x, (C,), w["norm_in_weight"], w["norm_in_bias"], eps=eps)
    x_in = xln
    g = F.linear(xln, w["p_in_weight"]) * F.linear(xln, w["g_in_weight"]).sigmoid()
    if mask is not None:
        g = g * mask.unsqueeze(-1)
    a, b = torch.chunk(g, 2, dim=-1)
    if direction == "outgoing":
        y = torch.einsum("bikd,bjkd->bijd", a, b)
    else:
        y = torch.einsum("bkid,bkjd->bijd", a, b)
    yln = F.layer_norm(y, y.shape[-1:], w["norm_out_weight"], w["norm_out_bias"], eps=eps)
    return F.linear(yln, w["p_out_weight"]) * F.linear(x_in, w["g_out_weight"]).sigmoid()


def torch_attn_pair_bias_qknorm(
    X,
    WQ,
    b_q,
    WK,
    WV,
    Z,
    W_ln,
    B_ln,
    W_proj_z,
    B_proj_z,
    W_proj_g,
    W_proj_o,
    H,
    D,
    *,
    qkn_w_q=None,
    qkn_b_q=None,
    qkn_w_k=None,
    qkn_b_k=None,
    key_mask=None,
    scale=1.0,
    eps=1e-5,
):
    """K-Fold attention-pair-bias WITH optional Q-bias and cross-head qk_norm.

    Extends torch_attn_pair_bias with:
      * a Q-projection bias `b_q` (K,V are bias-free), and
      * qk_norm = a LayerNorm over the FULL H*D channel applied to Q and K *before*
        the head split (matches transformers.py AttentionPairBias, use_qk_norm=True:
        Linear -> LayerNorm(channel_a) -> head-split). Full LayerNorm: mean-center +
        var + eps + affine (gamma, beta). When qkn_* are None, no qk_norm.

    Weights are matmul-convention (in, out). Gate/output projections are bias-free
    (K-Fold linear_g / linear_out are LinearNoBias). Returns (M, H*D).
    """
    M, _Cs = X.shape  # _Cs = H*D input channel
    Cz = Z.shape[-1]
    q = X @ WQ + b_q  # (M, H*D)
    k = X @ WK  # (M, H*D)
    v = X @ WV  # (M, H*D)
    if qkn_w_q is not None:  # cross-head LayerNorm over H*D, BEFORE head split
        q = F.layer_norm(q, (H * D,), qkn_w_q, qkn_b_q, eps=eps)
        k = F.layer_norm(k, (H * D,), qkn_w_k, qkn_b_k, eps=eps)
    q = q.view(M, H, D).permute(1, 0, 2)  # (H, M, D)
    k = k.view(M, H, D).permute(1, 0, 2)
    v = v.view(M, H, D).permute(1, 0, 2)
    bp = (F.layer_norm(Z, (Cz,), W_ln, B_ln, eps=eps) @ W_proj_z + B_proj_z).permute(2, 0, 1)
    scores = torch.matmul(q * scale, k.transpose(-1, -2)) + bp
    if key_mask is not None:  # mask padded KEY positions (..,Lk) -> -inf (keep scores' dtype)
        scores = scores.masked_fill(~key_mask.bool()[None, None, :], float("-inf"))
    o = torch.matmul(torch.softmax(scores, -1), v).permute(1, 0, 2).reshape(M, H * D)
    g = torch.sigmoid(F.linear(X, W_proj_g))
    return F.linear(g * o, W_proj_o)
