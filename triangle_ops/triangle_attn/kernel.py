"""Fused triangle-attention Triton kernels (low-level) — two-launch "v3" winner.

Triangle attention's bias Bp[h,q,k] = LN(x[q,k,:]) @ W_proj_z[:,h] is SHARED
across all N rows i.  A single fused launch would recompute it per row
(O(N³·H·C_in)); instead we split:

  bias_proj_kernel     — materialize Bp (H, N, N) ONCE  (O(N²·H·C_in))
  triangle_attn_kernel — per (i, h, q_block): LN + Q/K/V proj + FA-v2 with Bp
                         as input.  K/V are produced by a SINGLE matmul through
                         a feature-interleaved W_KV + tl.split, and BLOCK_M is
                         allowed up to 128 to amortize K/V loads across queries.

LN affine is pre-folded into Q/K/V and bias-proj weights (see module.py).
gate + Wo stay in PyTorch.  Fusion scope: LN + Q/K/V proj + bias proj + attention.

Naming follows the triangle_mul kernel style: `_stride{0,1,2}` for strides,
`_offs` for index ranges, `_fp32` for float casts, `_mask` for masks, `Out` tile.
"""

import triton
import triton.language as tl
import torch

from .._common.dtype import tl_io_dtype


# ===========================================================================
# Kernel 1: bias projection  (LN(x) + W_proj_z → Bp (H, N, N), materialized once)
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_QK': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_QK': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_QK': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_QK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_QK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_QK': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_QK': 256}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C_in', 'H'],
)
@triton.jit
def bias_proj_kernel(
    X_ptr, X_stride0, X_stride1, X_stride2,
    WZ_ptr, WZ_stride0, WZ_stride1,
    sWZ_ptr, sWZ_stride0,
    BZc_ptr, BZc_stride0,
    Bias_ptr, Bias_stride0, Bias_stride1, Bias_stride2,
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    BLOCK_QK: tl.constexpr,
):
    # Grid is (qk_blocks, H) so qk_blocks (large for big N) uses x dim,
    # avoiding the 65535 CUDA y-dim limit at N ≥ 1536 with small BLOCK_QK.
    pid_qk = tl.program_id(0)
    pid_h = tl.program_id(1)

    qk_offs = pid_qk * BLOCK_QK + tl.arange(0, BLOCK_QK)
    c_offs = tl.arange(0, C_in)
    qk_mask = qk_offs < (N * N)

    q_idx = qk_offs // N
    k_idx = qk_offs % N

    X_tile = tl.load(
        X_ptr + q_idx[:, None] * X_stride0 + k_idx[:, None] * X_stride1 + c_offs[None, :] * X_stride2,
        mask=qk_mask[:, None], other=0.0,
    )
    X_fp32 = X_tile.to(tl.float32)

    inv_C = 1.0 / C_in
    mean = tl.sum(X_fp32, axis=-1) * inv_C
    diff = X_fp32 - mean[:, None]
    var = tl.sum(diff * diff, axis=-1) * inv_C
    rstd = 1.0 / tl.sqrt(var + EPS)

    w_z = tl.load(WZ_ptr + pid_h * WZ_stride0 + c_offs * WZ_stride1)
    sWZ_h = tl.load(sWZ_ptr + pid_h * sWZ_stride0)
    BZc_h = tl.load(BZc_ptr + pid_h * BZc_stride0)

    bias_dot = tl.sum(X_fp32 * w_z[None, :], axis=-1)
    bias = rstd * (bias_dot - mean * sWZ_h) + BZc_h

    tl.store(
        Bias_ptr + pid_h * Bias_stride0 + q_idx * Bias_stride1 + k_idx * Bias_stride2,
        bias.to(IO_DTYPE),
        mask=qk_mask,
    )


# ===========================================================================
# Kernel 2: triangle attention with K/V concat (+ tl.split)
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C_in', 'H', 'D'],
)
@triton.jit
def triangle_attn_kernel(
    X_ptr, X_stride0, X_stride1, X_stride2,
    WQ_ptr, WQ_stride0, WQ_stride1,           # (C_in, H*D)
    WKV_ptr, WKV_stride0, WKV_stride1,        # (C_in, 2*H*D) feature-interleaved per head
    sWQ_ptr, sWQ_stride0,
    sWK_ptr, sWK_stride0,
    sWV_ptr, sWV_stride0,
    BQc_ptr, BQc_stride0,
    BKc_ptr, BKc_stride0,
    BVc_ptr, BVc_stride0,
    Bias_ptr, Bias_stride0, Bias_stride1, Bias_stride2,    # (H, N, N)
    Mask_ptr, Mask_stride0, Mask_stride1,
    O_ptr, O_stride0, O_stride1, O_stride2,   # (N, N, H*D)
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    NEG_INF: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    kv_cols = pid_h * 2 * D + tl.arange(0, 2 * D)
    c_offs = tl.arange(0, C_in)
    head_col = pid_h * D + d_offs
    q_mask = q_offs < N

    inv_C = 1.0 / C_in

    X_q = tl.load(
        X_ptr + pid_n * X_stride0 + q_offs[:, None] * X_stride1 + c_offs[None, :] * X_stride2,
        mask=q_mask[:, None], other=0.0,
    )
    X_q_fp32 = X_q.to(tl.float32)
    mean_q = tl.sum(X_q_fp32, axis=-1) * inv_C
    diff_q = X_q_fp32 - mean_q[:, None]
    var_q = tl.sum(diff_q * diff_q, axis=-1) * inv_C
    rstd_q = 1.0 / tl.sqrt(var_q + EPS)

    WQ_h = tl.load(WQ_ptr + c_offs[:, None] * WQ_stride0 + head_col[None, :] * WQ_stride1)
    WKV_h = tl.load(WKV_ptr + c_offs[:, None] * WKV_stride0 + kv_cols[None, :] * WKV_stride1)
    sWQ_h = tl.load(sWQ_ptr + head_col * sWQ_stride0)
    sWK_h = tl.load(sWK_ptr + head_col * sWK_stride0)
    sWV_h = tl.load(sWV_ptr + head_col * sWV_stride0)
    BQc_h = tl.load(BQc_ptr + head_col * BQc_stride0)
    BKc_h = tl.load(BKc_ptr + head_col * BKc_stride0)
    BVc_h = tl.load(BVc_ptr + head_col * BVc_stride0)

    Q_acc = tl.dot(X_q, WQ_h)
    Q = rstd_q[:, None] * (Q_acc - mean_q[:, None] * sWQ_h[None, :]) + BQc_h[None, :]
    Q_scaled = (Q * SCALE).to(IO_DTYPE)

    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        X_k = tl.load(
            X_ptr + pid_n * X_stride0 + k_offs[:, None] * X_stride1 + c_offs[None, :] * X_stride2,
            mask=k_mask[:, None], other=0.0,
        )
        X_k_fp32 = X_k.to(tl.float32)
        mean_k = tl.sum(X_k_fp32, axis=-1) * inv_C
        diff_k = X_k_fp32 - mean_k[:, None]
        var_k = tl.sum(diff_k * diff_k, axis=-1) * inv_C
        rstd_k = 1.0 / tl.sqrt(var_k + EPS)

        KV_acc = tl.dot(X_k, WKV_h)                                                    # (BLOCK_K, 2*D)
        KV_3d = tl.reshape(KV_acc, (BLOCK_K, D, 2))
        K_acc, V_acc = tl.split(KV_3d)                                                 # each (BLOCK_K, D)
        K_block = (rstd_k[:, None] * (K_acc - mean_k[:, None] * sWK_h[None, :]) + BKc_h[None, :]).to(IO_DTYPE)
        V_block = (rstd_k[:, None] * (V_acc - mean_k[:, None] * sWV_h[None, :]) + BVc_h[None, :]).to(IO_DTYPE)

        S = tl.dot(Q_scaled, tl.trans(K_block)).to(tl.float32)

        bias_tile = tl.load(
            Bias_ptr + pid_h * Bias_stride0 + q_offs[:, None] * Bias_stride1 + k_offs[None, :] * Bias_stride2,
            mask=(q_mask[:, None] & k_mask[None, :]), other=0.0,
        ).to(tl.float32)
        S = S + bias_tile

        if HAS_MASK:
            mask_row = tl.load(
                Mask_ptr + pid_n * Mask_stride0 + k_offs * Mask_stride1,
                mask=k_mask, other=0,
            )
            mask_bias = tl.where(mask_row != 0, 0.0, NEG_INF)
            S = S + mask_bias[None, :]
        S = tl.where(k_mask[None, :], S, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(P, axis=1)
        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(IO_DTYPE), V_block).to(tl.float32)
        m_i = m_new

    Out = O_acc / l_i[:, None]
    o_col = pid_h * D + d_offs
    tl.store(
        O_ptr + pid_n * O_stride0 + q_offs[:, None] * O_stride1 + o_col[None, :] * O_stride2,
        Out.to(IO_DTYPE),
        mask=q_mask[:, None],
    )


def triangle_attn_forward(
    X,
    WQ_c, sWQ, BQ_const,
    WKV_c, sWK, sWV, BK_const, BV_const,
    WZ_c, sWZ, BZ_const,
    O,
    scale=1.0, eps=1e-5, mask=None, Bias_buf=None,
):
    """Launch bias_proj_kernel + triangle_attn_kernel; write into O (N, N, H*D)."""
    N, N2, C_in = X.shape
    assert N == N2
    H = sWZ.shape[0]
    D = WQ_c.shape[1] // H
    assert WKV_c.shape == (C_in, 2 * H * D)
    assert O.shape == (N, N, H * D)

    io_dtype = tl_io_dtype(X.dtype)
    if Bias_buf is None:
        Bias_buf = torch.empty(H, N, N, device=X.device, dtype=X.dtype)

    grid_b = lambda meta: (triton.cdiv(N * N, meta['BLOCK_QK']), H)
    bias_proj_kernel[grid_b](
        X, X.stride(0), X.stride(1), X.stride(2),
        WZ_c, WZ_c.stride(0), WZ_c.stride(1),
        sWZ, sWZ.stride(0),
        BZ_const, BZ_const.stride(0),
        Bias_buf, Bias_buf.stride(0), Bias_buf.stride(1), Bias_buf.stride(2),
        N=N, C_in=C_in, H=H, EPS=eps, IO_DTYPE=io_dtype,
    )

    has_mask = mask is not None
    if has_mask:
        mask_i8 = mask.to(torch.int8).contiguous()
        mask_ptr, mask_stride0, mask_stride1 = mask_i8, mask_i8.stride(0), mask_i8.stride(1)
    else:
        mask_ptr, mask_stride0, mask_stride1 = X, 0, 0

    grid_a = lambda meta: (N, H, triton.cdiv(N, meta['BLOCK_M']))
    triangle_attn_kernel[grid_a](
        X, X.stride(0), X.stride(1), X.stride(2),
        WQ_c, WQ_c.stride(0), WQ_c.stride(1),
        WKV_c, WKV_c.stride(0), WKV_c.stride(1),
        sWQ, sWQ.stride(0),
        sWK, sWK.stride(0),
        sWV, sWV.stride(0),
        BQ_const, BQ_const.stride(0),
        BK_const, BK_const.stride(0),
        BV_const, BV_const.stride(0),
        Bias_buf, Bias_buf.stride(0), Bias_buf.stride(1), Bias_buf.stride(2),
        mask_ptr, mask_stride0, mask_stride1,
        O, O.stride(0), O.stride(1), O.stride(2),
        N=N, C_in=C_in, H=H, D=D,
        SCALE=scale, EPS=eps, NEG_INF=-1e9,
        HAS_MASK=has_mask, IO_DTYPE=io_dtype,
    )
    return O
