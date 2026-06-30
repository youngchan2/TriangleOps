"""Shared test fixtures + correctness ground truth = cuequivariance kernels.

Correctness is checked against cuequivariance's OPTIMIZED kernels — the same ones
K-Fold deploys — NOT a hand-written torch reference.  (An earlier torch reference
silently re-encoded an implementation bug in the gate input — `σ(W_g·x)` on the
raw x instead of `LN(x)` — so it could never catch that bug.  cuequiv is the
authoritative reference; the cuequiv wrappers live in `benchmarks/references.py`.)

Notes on the cuequiv ground truth per op:
  * attn_pair_bias / triangle_mul : cuequiv exposes the WHOLE fused op (gate
    included), so the comparison is end-to-end against cuequiv.
  * triangle_attn : cuequiv's `triangle_attention` is the attention CORE only
    (q,k,v,bias -> o).  LN / Q·K·V proj / gate / Wo live in K-Fold's torch wrapper
    (`MultiHeadAttention`), which gates on q_x = LN(x).  The reference therefore
    routes the core through cuequiv and mirrors K-Fold's torch epilogue exactly.
"""

import os
import sys

import pytest
import torch

# the cuequiv-backed references live in the sibling `benchmarks` package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cuequivariance_torch  # noqa: F401  (cuequiv kernels are the ground truth)

    from benchmarks.references import (
        cueq_attn_pair_bias,
        cueq_triangle_attention,
        cueq_triangle_multiplicative_update,
    )

    _HAS_CUEQ = True
    _CUEQ_ERR = None
except Exception as exc:  # cuequivariance not installed in this env
    _HAS_CUEQ = False
    _CUEQ_ERR = repr(exc)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
requires_cueq = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_CUEQ),
    reason=f"cuequivariance required for correctness ground truth ({_CUEQ_ERR})",
)


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def rnd(*shape, dtype, device, sd=0.05, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=device, dtype=dtype) * sd


# Per-dtype absolute tolerance vs the cuequivariance kernel.  Both sides are
# bf16/fp16 approximations of the same fp32 math; measured cross-kernel agreement
# is well inside these (bf16 max|Δ| ~2e-2, fp16 ~4e-3, cos > 0.99998).
TOL = {torch.bfloat16: 5e-2, torch.float16: 1e-2}


# --------------------------------------------------------------------------- #
# References — thin delegators to the cuequivariance kernels (ground truth)
# --------------------------------------------------------------------------- #
def ref_attn_pair_bias(*args, **kwargs):
    return cueq_attn_pair_bias(*args, **kwargs)


def ref_triangle_attn(*args, **kwargs):
    return cueq_triangle_attention(*args, **kwargs)


def ref_triangle_mul(x, mask, direction, eps, **w):
    return cueq_triangle_multiplicative_update(x, direction=direction, mask=mask, eps=eps, **w)
