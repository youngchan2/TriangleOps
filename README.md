# TriangleOps

Fused Triton kernels for the three AlphaFold-3 **Pairformer** primitives, tuned to
beat NVIDIA [`cuequivariance`](https://github.com/NVIDIA/cuEquivariance) on H100 (bf16) while being a drop-in replacement:

| op | fused scope | speedup vs cuequivariance (H100, bf16) |
|----|-------------|----------------------------------------|
| `attention_pair_bias` | QKV proj + LN(Z) + bias proj + attention | **1.7–4.1×** across all M (128–2048) |
| `triangle_multiplicative_update` | LN + linear + gated proj | **1.2–2.5×** across all L (128–2048) |
| `triangle_attention` | Q/K/V proj + bias proj + attention + gate (reads pre-normalized x̃) | **1.3–2.0×** across all N (128–2048) |

`triangle_mul` and `attention_pair_bias` fold the LayerNorm affine into the next
projection (fp32); `triangle_attention` instead computes `x̃ = LN(x)` once and shares it
across its kernels. All outputs **match cuequiv's own kernels** within bf16/fp16 rounding
(bf16 max|Δ| ≈ 2e-2, cos > 0.99998) — correctness is tested against cuequiv as the ground
truth, not a hand-written reference.

## Benchmarks (H100 PCIe, bf16)

Each plot: **left** = latency vs sequence length (log-log), **right** = speedup vs
cuequivariance (>1 = TriangleOps faster). Latency is the per-call deployment cost — the
LayerNorm-affine folding is a one-time model-init step (like BN-fusion / weight
prepacking), not a per-call cost. Reproduce with
`python -m benchmarks.sweep --op <op> --out-dir assets` (writes `assets/<op>.png`).

### `attention_pair_bias`  (M = single-seq length; H=4, D=32, C_z=128)
![attention_pair_bias](assets/attn_pair_bias.png)

| M | cueq (ms) | ours (ms) | speedup |
|---|-----------|-----------|---------|
| 128  | 0.324 | 0.079 | **4.11×** |
| 512  | 0.396 | 0.181 | **2.19×** |
| 1024 | 0.944 | 0.502 | **1.88×** |
| 2048 | 3.170 | 1.843 | **1.72×** |

→ faster than cuequiv at **every** size (1.7–4.1×).

### `triangle_multiplicative_update`  (L = pair-seq length; D=128, outgoing)
![triangle_multiplicative_update](assets/triangle_mul.png)

| L | cueq (ms) | ours (ms) | speedup |
|---|-----------|-----------|---------|
| 128  | 0.353 | 0.142 | **2.49×** |
| 512  | 0.694 | 0.561 | **1.24×** |
| 1024 | 2.952 | 2.416 | **1.22×** |
| 2048 | 13.800 | 11.485 | **1.20×** |

→ faster at every size (1.2–2.5×).

### `triangle_attention`  (N = pair-seq length; H=4, D=32, C_in=128)
![triangle_attention](assets/triangle_attn.png)

| N | cueq (ms) | ours (ms) | speedup |
|---|-----------|-----------|---------|
| 128  | 0.296   | 0.147 | **2.01×** |
| 256  | 0.545   | 0.318 | **1.72×** |
| 512  | 2.608   | 1.793 | **1.45×** |
| 768  | 6.895   | 5.360 | **1.29×** |
| 1024 | 15.06   | 11.68 | **1.29×** |
| 2048 | 118.95  | 84.35 | **1.41×** |

→ faster at **every** size (1.3–2.0×), ~2× at small N and still 1.3–1.4× at large N —
**no crossover**. (Earlier versions lost from N ≈ 768; moving LayerNorm out of the kernels
— `x̃ = LN(x)` computed once and shared, so the kernels drop per-token Welford/LN-fold —
recovered the large-N margin. The gate runs on `x̃` matching K-Fold. The row-shared bias
is still materialized once; fusing it on-chip is the remaining headroom.)

## What's fused — vs cuequivariance

Both use Triton; we just pack more stages into each launch (and fold LayerNorm into the
next projection's weights). `[ … ]` = one kernel/launch.

```text
attention_pair_bias
  cuequiv :  [QKV proj] [LN(z)+bias] [attention] [gate+Wo]          (cuDNN SDPA, ~4 launches)
  ours    :  [ QKV proj + LN(z) + bias + attention ]  [gate+Wo]     (1 Triton kernel + torch)

triangle_multiplicative_update
  cuequiv :  [LN] [gated proj] [einsum] [LN] [gated proj]           (5 launches)
  ours    :  [ LN + gated proj ] [einsum] [ LN + gated proj ]       (3 launches)

triangle_attention
  cuequiv :  [LN] [Q/K/V proj] [bias proj] [attention] [gate+Wo]    (cuDNN SDPA; rest eager)
  ours    :  x̃=LN(x) → [ bias ] [ Q/K/V + attention ] [ gate+Wo ]   (LN once in torch; 3 Triton kernels)
```

So: for `triangle_mul`/`attention_pair_bias` LN never gets its own kernel (folded into the
next proj); `triangle_attention` computes `x̃ = LN(x)` once in torch and its kernels read it.
`attention_pair_bias` inlines QKV + bias into the attention kernel (bias on-chip, 1 launch);
`triangle_attention` materializes the row-shared bias once, then Q/K/V+attention, then a
fused gate+Wo kernel (3 Triton launches, all reading x̃); the einsum (`triangle_mul`) stays
cuBLAS in both.

## Layout

```
triangle_ops/                 # the library (no cuequivariance dependency)
├── _common/                  #   shared: LN-affine absorption, weight layouts, dtype
├── attn_pair_bias/           #   kernel.py (@triton.jit + launch) · module.py (public API)
├── triangle_attn/
└── triangle_mul/
tests/                        # correctness vs pure-torch fp32 reference (pytest)
benchmarks/                   # references.py (cueq+torch) · harness.py · sweep.py
assets/<op>.png               # committed benchmark figures (shown above)
results/                      # ad-hoc generated CSV/PNG (gitignored)
```

Three layers: **kernel.py** (raw Triton) → **module.py** (dtype/precompute/public API) →
**tests + benchmarks** (consumers). The library never imports cuequivariance — that lives
only in `benchmarks/references.py`.

## Install / use

```bash
pip install -e .            # or add TriangleOps/ to PYTHONPATH
```

Each op exposes three entry points:

```python
import triangle_ops

# 1) one-shot — simplest (folds weights inside the call)
out = triangle_ops.triangle_multiplicative_update(
    x, direction="outgoing", mask=mask,
    norm_in_weight=..., norm_in_bias=..., p_in_weight=..., g_in_weight=...,
    norm_out_weight=..., norm_out_bias=..., p_out_weight=..., g_out_weight=...)

# 2) amortized — fold weights once at model init, fast path per call
pre = triangle_ops.triangle_mul.precompute(norm_in_weight, ..., g_out_weight)
out = triangle_ops.triangle_mul.forward(x, pre, direction="outgoing", mask=mask)
```

`triangle_ops.attention_pair_bias` / `triangle_ops.triangle_attention` follow the same
`precompute` / `forward` / one-shot pattern (signatures mirror cuequivariance + K-Fold).

## Design lever: LayerNorm-affine absorption

A LayerNorm followed by a linear projection is collapsed by folding the LN affine into
the projection weight (`_common/ln_absorption.py`):

```
(LN(x) @ Wᵀ)[o] = rstd·(Σ_i x[i]·Wc[o,i] − mean·ΣW[o]) + Bc[o]
    Wc = w_ln·W,  ΣW = Σ Wc,  Bc = Σ b_ln·W      # precomputed once at model init
```

so the fused kernel produces (mean, var, projection) in one Welford pass — no separate
LN kernel/buffer. **Used by `triangle_mul` and `attention_pair_bias`.** `triangle_attention`
instead computes `x̃ = LN(x)` once outside and its kernels read it directly (no absorption,
no per-token Welford) — this is what recovered its large-N speedup. Per-op kernel specifics
are in each `kernel.py` docstring; the headline optimizations were: D-major einsum layout
(triangle_mul), bias materialized once (`triangle_attention`, 3 launches: bias / Q·K·V+
attention / gate+Wo), K/V concat + larger BLOCK_M (triangle_attn), and load-x-once
internal-k-loop (triangle_mul Kernel A).

## Tests & benchmarks

```bash
# correctness — 60 cases (dtype × size × direction × mask) vs the cuequiv kernel
# as ground truth (needs the `bench` extra: cuequivariance)
CUDA_VISIBLE_DEVICES=1 python -m pytest tests/ -q

# latency sweep vs cuequiv + torch  (needs the `bench` extra: cuequivariance, matplotlib)
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.sweep --op triangle_mul --direction outgoing
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.sweep --op triangle_attn
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.sweep --op attn_pair_bias
```

The sweep reports per-call latency (`triangle_ops`) vs cuequiv + torch; the LN-affine
fold is a one-time model-init step, not counted per call.

> cuequiv falls back to plain torch below a sequence-length threshold (eager mode):
> `triangle_mul` L≤100 and `attention_pair_bias` M≤100 (pair size M²≤10000). All
> benchmark points at/above 128 compare against its **optimized** kernels.

The research history (v1/v2/v3 variants, profiling, rejected fusions) lives in the
repo's `bench/` tree; TriangleOps ships only the winning kernel per op.
