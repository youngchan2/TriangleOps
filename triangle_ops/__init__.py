"""TriangleOps — fused Triton kernels for AlphaFold-3 Pairformer primitives.

Three drop-in-faster replacements for cuequivariance (H100, bf16), each fusing
LayerNorm + projection + the op body, with the LN affine algebraically folded
into the projection weights:

    attention_pair_bias            — QKV proj + LN(Z) + bias proj + attention
    triangle_attention             — LN + Q/K/V proj + bias proj + attention
    triangle_multiplicative_update — LN + gated proj + triangular einsum + gated proj

Each op module exposes:
    precompute(...)  — one-time (model-init) weight fusion → opaque dict
    forward(x, pre, ...) — per-call fast path (precompute EXCLUDED / amortized)
    <op_name>(...)   — one-shot convenience (precompute INCLUDED)
"""

from . import attn_pair_bias, triangle_attn, triangle_mul
from .attn_pair_bias import attention_pair_bias
from .triangle_attn import triangle_attention
from .triangle_mul import triangle_multiplicative_update

__version__ = "0.1.0"

__all__ = [
    "attn_pair_bias",
    "triangle_attn",
    "triangle_mul",
    "attention_pair_bias",
    "triangle_attention",
    "triangle_multiplicative_update",
]
