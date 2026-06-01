"""Public API for fused attention-pair-bias.

Scope of the fused kernel: QKV projection + LN(Z) + bias projection + attention.
gate + output projection ("MLP after attention") stay in PyTorch — same split as
cuequivariance — because they are a small, already well-tuned matmul.

Typical usage (amortized): call `precompute(...)` ONCE at model init, then
`forward(...)` each step.  `attention_pair_bias(...)` bundles both for one-shot use.
"""

import torch

from .._common.ln_absorption import absorb_ln_matmul
from .._common.layouts import interleave_qkv
from .kernel import apb_forward


def precompute(WQ, WK, WV, W_ln, B_ln, W_proj_z, B_proj_z, H, D):
    """One-time (model-init) fusion of weights.

    Returns a dict consumed by `forward`:
        W_QKV          (N, 3*H*D)  per-head concat of Q/K/V
        W_combined     (H, N) fp32 LN-folded bias-projection weight
        sum_W_combined (H,)   fp32
        B_const        (H,)   fp32
    """
    W_QKV = interleave_qkv(WQ, WK, WV, H, D)
    # W_proj_z is (N, H) matmul-convention (z_norm @ W_proj_z); fold LN, keep fp32.
    W_combined, sum_W_combined, B_const = absorb_ln_matmul(
        W_proj_z, W_ln, B_ln, weight_dtype=torch.float32,
    )
    W_combined = W_combined.t().contiguous()       # (N, H) -> (H, N) for the kernel
    B_const = (B_const + B_proj_z.float()).contiguous()
    return {
        "W_QKV": W_QKV,
        "W_combined": W_combined,
        "sum_W_combined": sum_W_combined,
        "B_const": B_const,
    }


def forward(X, Z, pre, *, scale=1.0, eps=1e-5,
            W_proj_g, B_proj_g, W_proj_o, B_proj_o):
    """Per-call forward using precomputed weights `pre` (from `precompute`).

    Returns out (M, C_s).  gate + Wo are applied in PyTorch.
    """
    M, _ = X.shape
    H = pre["sum_W_combined"].shape[0]
    D = pre["W_QKV"].shape[1] // (3 * H)
    Og = torch.empty(M, H * D, device=X.device, dtype=X.dtype)
    apb_forward(X, pre["W_QKV"], Z, pre["W_combined"], pre["sum_W_combined"],
                pre["B_const"], Og, scale=scale, eps=eps)
    g = torch.sigmoid(torch.nn.functional.linear(X, W_proj_g, B_proj_g))
    return torch.nn.functional.linear(g * Og, W_proj_o, B_proj_o)


def attention_pair_bias(
    X, WQ, WK, WV, Z,
    W_ln, B_ln, W_proj_z, B_proj_z,
    W_proj_g, B_proj_g, W_proj_o, B_proj_o,
    H, D, scale=1.0, eps=1e-5,
):
    """End-to-end one-shot (precompute INCLUDED). For amortized latency, call
    `precompute` once and `forward` each step."""
    pre = precompute(WQ, WK, WV, W_ln, B_ln, W_proj_z, B_proj_z, H, D)
    return forward(X, Z, pre, scale=scale, eps=eps,
                   W_proj_g=W_proj_g, B_proj_g=B_proj_g,
                   W_proj_o=W_proj_o, B_proj_o=B_proj_o)
