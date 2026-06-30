"""Public API for fused triangle attention (AlphaFold-3 starting/ending node).

Fused scope: Q/K/V projection + bias projection + attention.  LN is NOT folded
into the projections; instead x̃ = LN(x) is materialized ONCE outside the kernels
and shared by the bias-proj, attention, and gate paths (mirrors cuequivariance).
gate + Wo (K-Fold's MultiHeadAttention._wrap_up) stay in PyTorch.

Mask follows K-Fold semantics: a pair-level (N, N) bool, broadcast over heads &
queries (each row i masks over its N key positions).
"""

import torch

from .._common.layouts import interleave_kv
from .kernel import fused_gate_forward, triangle_attn_forward


def precompute(W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z, H, D):
    """One-time (model-init) layout prep. WQ/WK/WV are (C_in, H*D), W_proj_z
    (C_in, H) (matmul convention).  LN is no longer absorbed — the kernels read a
    pre-normalized x̃.  Returns a dict consumed by `forward`."""
    WQ_c = WQ.contiguous()  # (C_in, H*D) matmul convention, no fold
    WKV_c = interleave_kv(WK, WV, H, D).contiguous()  # (C_in, 2*H*D), no fold
    WZ_c = W_proj_z.t().contiguous()  # (C_in, H) -> (H, C_in) for the bias kernel

    # bias-proj bias as an (H,) fp32 tensor (K-Fold's bias-proj is LinearNoBias).
    if B_proj_z is None:
        BZ = torch.zeros(H, device=WQ.device, dtype=torch.float32)
    else:
        BZ = B_proj_z.float().contiguous()

    return {
        "WQ_c": WQ_c,
        "WKV_c": WKV_c,
        "WZ_c": WZ_c,
        "BZ": BZ,
        # x̃ = LN(x) is computed once in `forward`; the kernels and the gate
        # epilogue all consume it, so the LN affine is needed here.
        "W_ln": W_ln,
        "B_ln": B_ln,
    }


def _wrap_up(x_gate_in, O_attn, W_proj_g, B_proj_g, W_proj_o, B_proj_o, H, D):
    """K-Fold MultiHeadAttention._wrap_up: gate(sigmoid linear) ⊙ O, then Wo."""
    g = torch.sigmoid(torch.nn.functional.linear(x_gate_in, W_proj_g, B_proj_g))
    g = g.view(g.shape[:-1] + (H, D))
    o = O_attn.view(O_attn.shape[:-1] + (H, D)) * g
    o = o.reshape(o.shape[:-2] + (H * D,))
    return torch.nn.functional.linear(o, W_proj_o, B_proj_o)


def forward(X, pre, *, mask=None, scale=1.0, eps=1e-5, W_proj_g, B_proj_g, W_proj_o, B_proj_o):
    """Per-call forward using precomputed weights `pre`.  X is (N, N, C_in);
    returns (N, N, C_in).  LN is computed ONCE as x̃ = LN(x) and shared by the
    attention/bias kernels and the gate epilogue (no LN-absorption)."""
    assert B_proj_g is None and B_proj_o is None, (
        "fused gate path assumes bias-free linear_g / linear_o (K-Fold LinearNoBias)"
    )
    N = X.shape[0]
    C_in = X.shape[-1]
    H = pre["WZ_c"].shape[0]
    D = pre["WQ_c"].shape[1] // H
    # x̃ = LN(x) computed once and shared by the kernels and the gate.
    X_ln = torch.nn.functional.layer_norm(X, (C_in,), pre["W_ln"].to(X.dtype), pre["B_ln"].to(X.dtype), eps)
    O_attn = torch.empty(N, N, H * D, device=X.device, dtype=X.dtype)
    triangle_attn_forward(
        X_ln,
        pre["WQ_c"],
        pre["WKV_c"],
        pre["WZ_c"],
        pre["BZ"],
        O_attn,
        scale=scale,
        mask=mask,
    )
    # Gate epilogue (fused): K-Fold gates on q_x = LN(x), so reuse x̃ directly.
    # fused_gate_forward allocates & returns Out (N, N, H*D); O_attn is contiguous
    # here so the (N,N,H*D) -> (N,N,H,D) view is free.
    return fused_gate_forward(O_attn.view(N, N, H, D), X_ln, W_proj_g, W_proj_o)


def triangle_attention(
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
    """End-to-end one-shot (precompute INCLUDED). For amortized latency, call
    `precompute` once and `forward` each step."""
    pre = precompute(W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z, H, D)
    return forward(
        X,
        pre,
        mask=mask,
        scale=scale,
        eps=eps,
        W_proj_g=W_proj_g,
        B_proj_g=B_proj_g,
        W_proj_o=W_proj_o,
        B_proj_o=B_proj_o,
    )
