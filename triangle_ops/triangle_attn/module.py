"""Public API for fused triangle attention (AlphaFold-3 starting/ending node).

Fused scope: LN(x) + Q/K/V projection + bias projection + attention.
gate + Wo (K-Fold's MultiHeadAttention._wrap_up) stay in PyTorch.

Mask follows K-Fold semantics: a pair-level (N, N) bool, broadcast over heads &
queries (each row i masks over its N key positions).
"""

import torch

from .._common.ln_absorption import absorb_ln_matmul
from .._common.layouts import interleave_kv
from .kernel import triangle_attn_forward


def precompute(W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z, H, D):
    """One-time (model-init) fusion. WQ/WK/WV are (C_in, H*D), W_proj_z (C_in, H)
    (matmul convention).  Returns a dict consumed by `forward`."""
    proj_dtype = WQ.dtype

    WQ_c, sWQ, BQ_const = absorb_ln_matmul(WQ, W_ln, B_ln, weight_dtype=proj_dtype)

    # K/V: fold LN (keep fp32 for the interleave), interleave per head, then cast.
    WK_c_f32, sWK, BK_const = absorb_ln_matmul(WK, W_ln, B_ln, weight_dtype=torch.float32)
    WV_c_f32, sWV, BV_const = absorb_ln_matmul(WV, W_ln, B_ln, weight_dtype=torch.float32)
    WKV_c = interleave_kv(WK_c_f32, WV_c_f32, H, D).to(proj_dtype).contiguous()

    # bias proj: fold LN, transpose to (H, C_in) for the kernel, add B_proj_z.
    WZ_c_f32, sWZ, BZ_const = absorb_ln_matmul(W_proj_z, W_ln, B_ln, weight_dtype=torch.float32)
    WZ_c = WZ_c_f32.t().contiguous()                       # (C_in, H) -> (H, C_in)
    BZ_const = (BZ_const + B_proj_z.float()).contiguous()

    return {
        "WQ_c": WQ_c, "sWQ": sWQ, "BQ_const": BQ_const,
        "WKV_c": WKV_c, "sWK": sWK, "sWV": sWV,
        "BK_const": BK_const, "BV_const": BV_const,
        "WZ_c": WZ_c, "sWZ": sWZ, "BZ_const": BZ_const,
    }


def _wrap_up(x_gate_in, O_attn, W_proj_g, B_proj_g, W_proj_o, B_proj_o, H, D):
    """K-Fold MultiHeadAttention._wrap_up: gate(sigmoid linear) ⊙ O, then Wo."""
    g = torch.sigmoid(torch.nn.functional.linear(x_gate_in, W_proj_g, B_proj_g))
    g = g.view(g.shape[:-1] + (H, D))
    o = O_attn.view(O_attn.shape[:-1] + (H, D)) * g
    o = o.reshape(o.shape[:-2] + (H * D,))
    return torch.nn.functional.linear(o, W_proj_o, B_proj_o)


def forward(X, pre, *, mask=None, scale=1.0, eps=1e-5,
            W_proj_g, B_proj_g, W_proj_o, B_proj_o):
    """Per-call forward using precomputed weights `pre`.  X is (N, N, C_in);
    returns (N, N, C_in)."""
    N = X.shape[0]
    H = pre["sWZ"].shape[0]
    D = pre["WQ_c"].shape[1] // H
    O_attn = torch.empty(N, N, H * D, device=X.device, dtype=X.dtype)
    triangle_attn_forward(
        X,
        pre["WQ_c"], pre["sWQ"], pre["BQ_const"],
        pre["WKV_c"], pre["sWK"], pre["sWV"], pre["BK_const"], pre["BV_const"],
        pre["WZ_c"], pre["sWZ"], pre["BZ_const"],
        O_attn, scale=scale, eps=eps, mask=mask,
    )
    return _wrap_up(X, O_attn, W_proj_g, B_proj_g, W_proj_o, B_proj_o, H, D)


def triangle_attention(
    X, W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z,
    W_proj_g, B_proj_g, W_proj_o, B_proj_o,
    H, D, scale=1.0, eps=1e-5, mask=None,
):
    """End-to-end one-shot (precompute INCLUDED). For amortized latency, call
    `precompute` once and `forward` each step."""
    pre = precompute(W_ln, B_ln, WQ, WK, WV, W_proj_z, B_proj_z, H, D)
    return forward(X, pre, mask=mask, scale=scale, eps=eps,
                   W_proj_g=W_proj_g, B_proj_g=B_proj_g,
                   W_proj_o=W_proj_o, B_proj_o=B_proj_o)
