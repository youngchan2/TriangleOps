"""Fused attention-pair-bias Triton kernel (low-level).

Single launch over grid (M/BLOCK_M, H): per (q_block, head) it does
QKV projection + LN(Z)-Welford + bias projection + FlashAttention-v2.
The LN affine is pre-folded into the bias-projection weight (see module.py),
so Welford's single pass over Z also accumulates the bias dot product.

Kernel body is the register-optimized "v2" winner (157→126 reg/thread vs the
first cut): per-head QKV weight concat, scale applied to S (not Q), and Welford
merge across chunks with the naive within-chunk m2 (no chunk_diff intermediate).
gate + Wo are intentionally left to PyTorch (see module.py) — they are a tiny,
already well-tuned (M, H*D)×(H*D, C_s) matmul that doesn't fit this grid.

Naming follows the triangle_mul kernel style: `_stride{0,1,2}` for strides,
`_offs` for index ranges, `_fp32` for float casts, `_mask` for masks, `Out` tile.
"""

import triton
import triton.language as tl
import torch

from .._common.dtype import tl_io_dtype


@triton.autotune(
    configs=[
        # ---- Small BLOCK_M (8) — max parallelism, min reg ----
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 16, 'N_BLOCK': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 16, 'N_BLOCK': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 16, 'N_BLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 32, 'N_BLOCK': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 32, 'N_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 64, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 64, 'N_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_K': 128, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        # ---- Medium BLOCK_M (16) — matches H100 SM count at M=512 ----
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16, 'N_BLOCK': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16, 'N_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16, 'N_BLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16, 'N_BLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32, 'N_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32, 'N_BLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32, 'N_BLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64, 'N_BLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64, 'N_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 128, 'N_BLOCK': 16}, num_warps=8, num_stages=2),
        # ---- Larger BLOCK_M (32) — more compute per program ----
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 16, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 16, 'N_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'N_BLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'N_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'N_BLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'N_BLOCK': 16}, num_warps=8, num_stages=2),
        # ---- Large BLOCK_M (64) ----
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 16, 'N_BLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'N_BLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'N_BLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'H', 'D'],
)
@triton.jit
def _apb_kernel(
    X_ptr, X_stride0, X_stride1,
    Wqkv_ptr, Wqkv_stride0, Wqkv_stride1,    # (N, 3*H*D) concat: per-head [Q | K | V]
    Z_ptr, Z_stride0, Z_stride1, Z_stride2,
    Wc_ptr, Wc_stride0, Wc_stride1,          # w_combined (H, N)
    SWc_ptr, SWc_stride0,                    # sum_w_combined (H,)
    Bc_ptr, Bc_stride0,                      # b_const (H,)
    Og_ptr, Og_stride0, Og_stride1,          # (M, H*D) attention output (pre-gate, pre-Wo)
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, D)
    c_offs = tl.arange(0, N)
    q_mask = q_offs < M

    qkv_base = 3 * pid_h * D
    head_col = pid_h * D + d_offs

    X_q = tl.load(
        X_ptr + q_offs[:, None] * X_stride0 + c_offs[None, :] * X_stride1,
        mask=q_mask[:, None],
        other=0.0,
    )

    WQ_h = tl.load(
        Wqkv_ptr + c_offs[:, None] * Wqkv_stride0 + (qkv_base + d_offs)[None, :] * Wqkv_stride1
    )
    WK_h = tl.load(
        Wqkv_ptr + c_offs[:, None] * Wqkv_stride0 + (qkv_base + D + d_offs)[None, :] * Wqkv_stride1
    )
    WV_h = tl.load(
        Wqkv_ptr + c_offs[:, None] * Wqkv_stride0 + (qkv_base + 2 * D + d_offs)[None, :] * Wqkv_stride1
    )

    Q_block = tl.dot(X_q, WQ_h)
    Q_fp16 = Q_block.to(IO_DTYPE)

    sum_w_combined_h = tl.load(SWc_ptr + pid_h * SWc_stride0)
    b_const_h = tl.load(Bc_ptr + pid_h * Bc_stride0)

    inv_N = 1.0 / N

    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for k_start in range(0, M, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < M

        X_k = tl.load(
            X_ptr + k_offs[:, None] * X_stride0 + c_offs[None, :] * X_stride1,
            mask=k_mask[:, None],
            other=0.0,
        )
        K_block = tl.dot(X_k, WK_h).to(IO_DTYPE)
        V_block = tl.dot(X_k, WV_h).to(IO_DTYPE)

        S = tl.dot(Q_fp16, tl.trans(K_block)).to(tl.float32) * SCALE

        count = 0.0
        mean = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        m2 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        bias_partial = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for n_start in range(0, N, N_BLOCK):
            nb_offs = n_start + tl.arange(0, N_BLOCK)
            nb_mask = nb_offs < N

            z_chunk = tl.load(
                Z_ptr
                + q_offs[:, None, None] * Z_stride0
                + k_offs[None, :, None] * Z_stride1
                + nb_offs[None, None, :] * Z_stride2,
                mask=(q_mask[:, None, None] & k_mask[None, :, None] & nb_mask[None, None, :]),
                other=0.0,
            ).to(tl.float32)

            w_c_chunk = tl.load(
                Wc_ptr + pid_h * Wc_stride0 + nb_offs * Wc_stride1,
                mask=nb_mask, other=0.0,
            )

            chunk_size = N_BLOCK + 0.0
            chunk_sum = tl.sum(z_chunk, axis=-1)
            chunk_sum_sq = tl.sum(z_chunk * z_chunk, axis=-1)
            chunk_mean = chunk_sum / chunk_size
            chunk_m2 = chunk_sum_sq - chunk_size * chunk_mean * chunk_mean
            chunk_m2 = tl.maximum(chunk_m2, 0.0)

            new_count = count + chunk_size
            delta = chunk_mean - mean
            mean = mean + delta * (chunk_size / new_count)
            m2 = m2 + chunk_m2 + delta * delta * (count * chunk_size / new_count)
            count = new_count

            bias_partial += tl.sum(z_chunk * w_c_chunk[None, None, :], axis=-1)

        var = m2 * inv_N
        rstd = 1.0 / tl.sqrt(var + EPS)
        bias_tile = rstd * (bias_partial - mean * sum_w_combined_h) + b_const_h

        S = S + bias_tile
        S = tl.where(k_mask[None, :], S, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_i = l_i * alpha + tl.sum(P, axis=1)
        O_acc = O_acc * alpha[:, None] + tl.dot(P.to(IO_DTYPE), V_block).to(tl.float32)
        m_i = m_new

    Out = O_acc / l_i[:, None]
    tl.store(
        Og_ptr + q_offs[:, None] * Og_stride0 + head_col[None, :] * Og_stride1,
        Out.to(IO_DTYPE),
        mask=q_mask[:, None],
    )


def apb_forward(X, W_QKV, Z, W_combined, sum_W_combined, B_const, Og, scale=1.0, eps=1e-5):
    """Launch the fused attention-pair-bias kernel into `Og` (M, H*D).

    Args:
        X:               (M, N)            single-sequence input
        W_QKV:           (N, 3*H*D)        per-head concat (see _common.interleave_qkv)
        Z:               (M, M, N)         pair representation (raw)
        W_combined:      (H, N)   fp32     LN-folded bias-proj weight
        sum_W_combined:  (H,)     fp32
        B_const:         (H,)     fp32
        Og:              (M, H*D)          output buffer (pre gate/Wo)
        scale:           float             attention scale (applied to S)
    """
    M, N = X.shape
    H = sum_W_combined.shape[0]
    D = W_QKV.shape[1] // (3 * H)
    assert W_QKV.shape == (N, 3 * H * D)
    assert Z.shape == (M, M, N)
    assert W_combined.shape == (H, N)
    assert B_const.shape == (H,)
    assert Og.shape == (M, H * D)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), H)
    _apb_kernel[grid](
        X, X.stride(0), X.stride(1),
        W_QKV, W_QKV.stride(0), W_QKV.stride(1),
        Z, Z.stride(0), Z.stride(1), Z.stride(2),
        W_combined, W_combined.stride(0), W_combined.stride(1),
        sum_W_combined, sum_W_combined.stride(0),
        B_const, B_const.stride(0),
        Og, Og.stride(0), Og.stride(1),
        M=M, N=N, H=H, D=D, SCALE=scale, EPS=eps,
        IO_DTYPE=tl_io_dtype(X.dtype),
    )
    return Og
