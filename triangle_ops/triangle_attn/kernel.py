"""Fused triangle-attention Triton kernels (low-level) — two-launch "v3" winner.

LN is computed ONCE outside the kernels as x̃ = LN(x) (see module.py) and the
kernels read that pre-normalized x̃ directly — there is no LN-absorption and no
per-token Welford inside the kernels.

Triangle attention's bias Bp[h,q,k] = x̃[q,k,:] @ W_proj_z[:,h] + B_proj_z[h] is
SHARED across all N rows i.  A single fused launch would recompute it per row
(O(N³·H·C_in)); instead we split:

  bias_proj_kernel     — materialize Bp (H, N, N) ONCE  (O(N²·H·C_in))
  triangle_attn_kernel — per (i, h, q_block): Q/K/V proj + FA-v2 with Bp
                         as input.  K/V are produced by a SINGLE matmul through
                         a feature-interleaved W_KV + tl.split, and BLOCK_M is
                         allowed up to 128 to amortize K/V loads across queries.

gate + Wo stay in PyTorch.  Fusion scope: Q/K/V proj + bias proj + attention
(reading the shared x̃).

Naming follows the triangle_mul kernel style: `_stride{0,1,2}` for strides,
`_offs` for index ranges, `_fp32` for float casts, `_mask` for masks, `Out` tile.
"""

import torch
import triton
import triton.language as tl

from .._common.dtype import tl_io_dtype


# ===========================================================================
# Kernel 1: bias projection  (LN(x) + W_proj_z → Bp (H, N, N), materialized once)
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_QK": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_QK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_QK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QK": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_QK": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_QK": 256}, num_warps=4, num_stages=2),
    ],
    key=["N", "C_in", "H"],
)
@triton.jit
def bias_proj_kernel(
    X_ptr,
    X_stride0,
    X_stride1,
    X_stride2,
    WZ_ptr,
    WZ_stride0,
    WZ_stride1,
    BZ_ptr,
    BZ_stride0,
    Bias_ptr,
    Bias_stride0,
    Bias_stride1,
    Bias_stride2,
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    BLOCK_QK: tl.constexpr,
):
    # X is the pre-normalized x̃ = LN(x); bias[h,q,k] = Σ_c x̃[q,k,c]·WZ[h,c] + BZ[h].
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
        mask=qk_mask[:, None],
        other=0.0,
    )
    X_fp32 = X_tile.to(tl.float32)

    w_z = tl.load(WZ_ptr + pid_h * WZ_stride0 + c_offs * WZ_stride1)
    BZ_h = tl.load(BZ_ptr + pid_h * BZ_stride0)

    bias = tl.sum(X_fp32 * w_z[None, :], axis=-1) + BZ_h

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
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["N", "C_in", "H", "D"],
)
@triton.jit
def triangle_attn_kernel(
    X_ptr,
    X_stride0,
    X_stride1,
    X_stride2,
    WQ_ptr,
    WQ_stride0,
    WQ_stride1,  # (C_in, H*D)
    WKV_ptr,
    WKV_stride0,
    WKV_stride1,  # (C_in, 2*H*D) feature-interleaved per head
    Bias_ptr,
    Bias_stride0,
    Bias_stride1,
    Bias_stride2,  # (H, N, N)
    Mask_ptr,
    Mask_stride0,
    Mask_stride1,
    O_ptr,
    O_stride0,
    O_stride1,
    O_stride2,  # (N, N, H*D)
    N: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    NEG_INF: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # X is the pre-normalized x̃ = LN(x); Q/K/V are plain projections of x̃.
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    kv_cols = pid_h * 2 * D + tl.arange(0, 2 * D)
    c_offs = tl.arange(0, C_in)
    head_col = pid_h * D + d_offs
    q_mask = q_offs < N

    X_q = tl.load(
        X_ptr + pid_n * X_stride0 + q_offs[:, None] * X_stride1 + c_offs[None, :] * X_stride2,
        mask=q_mask[:, None],
        other=0.0,
    )

    WQ_h = tl.load(WQ_ptr + c_offs[:, None] * WQ_stride0 + head_col[None, :] * WQ_stride1)
    WKV_h = tl.load(WKV_ptr + c_offs[:, None] * WKV_stride0 + kv_cols[None, :] * WKV_stride1)

    Q_acc = tl.dot(X_q, WQ_h)
    Q_scaled = (Q_acc * SCALE).to(IO_DTYPE)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        X_k = tl.load(
            X_ptr + pid_n * X_stride0 + k_offs[:, None] * X_stride1 + c_offs[None, :] * X_stride2,
            mask=k_mask[:, None],
            other=0.0,
        )

        KV_acc = tl.dot(X_k, WKV_h)  # (BLOCK_K, 2*D)
        KV_3d = tl.reshape(KV_acc, (BLOCK_K, D, 2))
        K_acc, V_acc = tl.split(KV_3d)  # each (BLOCK_K, D)
        K_block = K_acc.to(IO_DTYPE)
        V_block = V_acc.to(IO_DTYPE)

        S = tl.dot(Q_scaled, tl.trans(K_block)).to(tl.float32)

        bias_tile = tl.load(
            Bias_ptr + pid_h * Bias_stride0 + q_offs[:, None] * Bias_stride1 + k_offs[None, :] * Bias_stride2,
            mask=(q_mask[:, None] & k_mask[None, :]),
            other=0.0,
        ).to(tl.float32)
        S = S + bias_tile

        if HAS_MASK:
            mask_row = tl.load(
                Mask_ptr + pid_n * Mask_stride0 + k_offs * Mask_stride1,
                mask=k_mask,
                other=0,
            )
            mask_bias = tl.where(mask_row != 0, 0.0, NEG_INF)
            S = S + mask_bias[None, :]
        S = tl.where(k_mask[None, :], S, -float("inf"))

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
    X_ln,
    WQ_c,
    WKV_c,
    WZ_c,
    BZ,
    out,
    scale=1.0,
    mask=None,
    Bias_buf=None,
):
    """Launch bias_proj_kernel + triangle_attn_kernel; write into out (N, N, H*D).
    X_ln is the pre-normalized x̃ = LN(x); the kernels read it directly (no LN)."""
    N, N2, C_in = X_ln.shape
    assert N == N2
    H = WZ_c.shape[0]
    D = WQ_c.shape[1] // H
    assert WKV_c.shape == (C_in, 2 * H * D)
    assert out.shape == (N, N, H * D)

    io_dtype = tl_io_dtype(X_ln.dtype)
    if Bias_buf is None:
        Bias_buf = torch.empty(H, N, N, device=X_ln.device, dtype=X_ln.dtype)

    grid_b = lambda meta: (triton.cdiv(N * N, meta["BLOCK_QK"]), H)
    bias_proj_kernel[grid_b](
        X_ln,
        X_ln.stride(0),
        X_ln.stride(1),
        X_ln.stride(2),
        WZ_c,
        WZ_c.stride(0),
        WZ_c.stride(1),
        BZ,
        BZ.stride(0),
        Bias_buf,
        Bias_buf.stride(0),
        Bias_buf.stride(1),
        Bias_buf.stride(2),
        N=N,
        C_in=C_in,
        H=H,
        IO_DTYPE=io_dtype,
    )

    has_mask = mask is not None
    if has_mask:
        mask_i8 = mask.to(torch.int8).contiguous()
        mask_ptr, mask_stride0, mask_stride1 = (
            mask_i8,
            mask_i8.stride(0),
            mask_i8.stride(1),
        )
    else:
        mask_ptr, mask_stride0, mask_stride1 = X_ln, 0, 0

    grid_a = lambda meta: (N, H, triton.cdiv(N, meta["BLOCK_M"]))
    triangle_attn_kernel[grid_a](
        X_ln,
        X_ln.stride(0),
        X_ln.stride(1),
        X_ln.stride(2),
        WQ_c,
        WQ_c.stride(0),
        WQ_c.stride(1),
        WKV_c,
        WKV_c.stride(0),
        WKV_c.stride(1),
        Bias_buf,
        Bias_buf.stride(0),
        Bias_buf.stride(1),
        Bias_buf.stride(2),
        mask_ptr,
        mask_stride0,
        mask_stride1,
        out,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=N,
        C_in=C_in,
        H=H,
        D=D,
        SCALE=scale,
        NEG_INF=-1e9,
        HAS_MASK=has_mask,
        IO_DTYPE=io_dtype,
    )
    return out


# ===========================================================================
# Kernel 3: fused gate epilogue  —  Out = (O * sigmoid(g_in @ Wg)) @ Wo
# Shared by triangle-attn wrap-up AND attention-pair-bias (trunk). The ONLY
# difference is how O is read (STRIDED_O constexpr):
#   STRIDED_O=True  : triangle wrap-up — O is (N, N, H, D), non-contiguous.
#                     decode row m -> (n1, n2), col c -> (h, d); gather at strides.
#   STRIDED_O=False : apb — O is (M, C) contiguous; read m*C + c directly.
# Everything else (g_in read, both matmuls, sigmoid, multiply, store) is shared.
# Compile-time branch -> two specialized kernels, zero runtime cost.
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=w, num_stages=s)
        for bm in (64, 128, 256, 512)
        for w in (4, 8)
        for s in (2, 3)
    ],
    key=["M", "C"],
)
@triton.jit
def fused_gate_kernel(
    O_ptr,
    O_sN1,
    O_sN2,
    O_sH,
    O_sD,  # strided-O args (used iff STRIDED_O)
    GIN_ptr,  # gate input (M, C) contiguous: LN(x) [tri] or ã [apb]
    WG_ptr,
    WO_ptr,  # (C, C) matmul-convention (already transposed)
    Out_ptr,  # (M, C) contiguous output
    M,
    N,  # N used iff STRIDED_O
    C: tl.constexpr,
    D: tl.constexpr,
    STRIDED_O: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    cols = tl.arange(0, C)

    if STRIDED_O:  # triangle wrap-up: O = (N, N, H, D), non-contiguous
        n1 = rows // N
        n2 = rows % N
        h = cols // D
        d = cols % D
        o_off = (n1[:, None] * O_sN1 + n2[:, None] * O_sN2) + (h[None, :] * O_sH + d[None, :] * O_sD)
        o = tl.load(O_ptr + o_off, mask=rmask[:, None], other=0.0)
    else:  # apb: O = (M, C) contiguous
        o = tl.load(O_ptr + rows[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0)

    gin = tl.load(GIN_ptr + rows[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0)
    wg = tl.load(WG_ptr + cols[:, None] * C + cols[None, :])
    wo = tl.load(WO_ptr + cols[:, None] * C + cols[None, :])

    g = tl.sigmoid(tl.dot(gin, wg))  # gate = sigmoid(g_in @ Wg)
    acc = tl.dot((o.to(tl.float32) * g).to(gin.dtype), wo)  # (O * g) @ Wo
    tl.store(Out_ptr + rows[:, None] * C + cols[None, :], acc.to(gin.dtype), mask=rmask[:, None])


def fused_gate_forward(o_attn, qx_ln, WG, WO):
    """Triangle wrap-up (STRIDED_O=True): Out = (o_attn * sigmoid(qx_ln @ Wg)) @ Wo.

    o_attn : (N, N, H, D)  attention output (any strides — gathered at native strides)
    qx_ln  : (N, N, C)     = LN(x), contiguous  (K-Fold gates on q_x = LN(x))
    WG, WO : (out, in)     K-Fold linear_g / linear_o weights; transposed to (in, out)
    returns: (N, N, C)     contiguous
    """
    N1, N2, H, D = o_attn.shape
    N, C, M = N1, H * D, N1 * N2
    out = torch.empty(N1, N2, C, device=o_attn.device, dtype=o_attn.dtype)
    Qf = qx_ln.reshape(M, C)  # free view (contiguous)
    Of = out.reshape(M, C)  # free view (contiguous)
    wg = WG.t().contiguous()  # (out,in) -> (in,out) so qx @ wg == linear_g(qx)
    wo = WO.t().contiguous()
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    fused_gate_kernel[grid](
        o_attn,
        o_attn.stride(0),
        o_attn.stride(1),
        o_attn.stride(2),
        o_attn.stride(3),
        Qf,
        wg,
        wo,
        Of,
        M,
        N,
        C=C,
        D=D,
        STRIDED_O=True,
    )
    return out


def apb_gate_forward(o_in, gate_in, WG, WO):
    """Attention-pair-bias gate epilogue (STRIDED_O=False, contiguous O):
        Out = (o_in * sigmoid(gate_in @ Wg)) @ Wo.

    o_in, gate_in : (..., C) contiguous  (gate_in = ã, the normed single rep)
    WG, WO        : (out, in)  K-Fold linear_g / linear_out weights; transposed to (in, out)
    returns       : same shape as o_in, contiguous.  (trunk apb only — no single-cond gate)
    """
    C = o_in.shape[-1]
    M = o_in.numel() // C
    out = torch.empty_like(o_in)
    Of = o_in.reshape(M, C)
    Gf = gate_in.reshape(M, C)
    Outf = out.reshape(M, C)
    wg = WG.t().contiguous()
    wo = WO.t().contiguous()
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    fused_gate_kernel[grid](
        Of,
        0,
        0,
        0,
        0,  # strided-O args unused (STRIDED_O=False)
        Gf,
        wg,
        wo,
        Outf,
        M,
        0,
        C=C,
        D=1,
        STRIDED_O=False,
    )
    return out
