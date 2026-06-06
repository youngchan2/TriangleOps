"""Unified sequence-length sweep for the three TriangleOps ops vs cuequiv + torch.

Run from the TriangleOps/ dir (so `triangle_ops` and `benchmarks` import):
    CUDA_VISIBLE_DEVICES=1 python -m benchmarks.sweep --op triangle_mul
    CUDA_VISIBLE_DEVICES=1 python -m benchmarks.sweep --op triangle_attn --seq-lens 128 256 512
    CUDA_VISIBLE_DEVICES=1 python -m benchmarks.sweep --op attn_pair_bias --no-torch

Writes results/<op>/sweep.csv and results/<op>/sweep.png.
'triangle_ops' = precompute EXCLUDED (deployment-amortized);
'triangle_ops_incl' = precompute INCLUDED (strict apples-to-apples vs cueq).
"""

import argparse
import csv
import os
import sys

import torch

from .harness import BUILDERS, time_ms

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_DEFAULT_SIZES = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
_SERIES = [
    ("triangle_ops", "TriangleOps (excl pc)", "#1f77b4", "-", "o"),
    ("triangle_ops_incl", "TriangleOps (incl pc)", "#1f77b4", "--", "^"),
    ("cueq", "cuequivariance", "#d62728", "-", "s"),
    ("torch", "torch (bf16)", "#ff7f0e", ":", "x"),
]

# parity (break-even vs cuequiv) on the speedup plot: a SOLID line plus a shaded
# "cuequiv is faster" band (speedup < 1) — so it can't be confused with the
# dashed incl-pc data series.
_PARITY = dict(color="#333333", linewidth=1.6, linestyle="-", zorder=1.5)
_LOSS_SHADE = dict(color="#d62728", alpha=0.07, zorder=0)


def make_plot(rows, op, dtype, out_dir):
    """Draw latency (left) + speedup-vs-cuequiv (right) → out_dir/{op}.png.
    Reused by the live sweep and by offline re-plotting from a saved CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["size"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: latency
    for k, lbl, c, ls, mk in _SERIES:
        ax1.plot(sizes, [r[k] for r in rows], label=lbl, color=c, linestyle=ls, marker=mk)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("seq length")
    ax1.set_ylabel("latency [ms]")
    ax1.set_title(f"{op} latency ({dtype})")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    # right: speedup vs cuequiv — shaded loss band + solid parity line
    ax2.axhspan(0.0, 1.0, **_LOSS_SHADE)
    for k, lbl, c, ls, mk in _SERIES:
        if not k.startswith("triangle_ops"):
            continue
        ax2.plot(
            sizes,
            [r["cueq"] / r[k] for r in rows],
            label=f"{lbl} / cueq",
            color=c,
            linestyle=ls,
            marker=mk,
        )
    ax2.axhline(1.0, label="parity (= cuequiv)", **_PARITY)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("seq length")
    ax2.set_ylabel("speedup vs cueq (>1 = faster)")
    ax2.set_title(f"{op} speedup vs cuequivariance")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{op}.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {png_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=list(BUILDERS), required=True)
    p.add_argument("--seq-lens", type=int, nargs="+", default=_DEFAULT_SIZES)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument(
        "--direction",
        choices=["outgoing", "incoming"],
        default="outgoing",
        help="triangle_mul only",
    )
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=12)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--no-torch", action="store_true")
    p.add_argument("--torch-max", type=int, default=1024, help="skip torch baseline above this size")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    out_dir = args.out_dir or os.path.join(ROOT, "results", args.op)
    os.makedirs(out_dir, exist_ok=True)
    builder = BUILDERS[args.op]

    keys = [s[0] for s in _SERIES]
    rows = []
    print(
        f"=== sweep op={args.op} dtype={args.dtype} dir={args.direction} "
        f"device={torch.cuda.get_device_name(device)} ==="
    )
    for size in args.seq_lens:
        it = args.iters if size <= 1024 else max(8, args.iters // 3)
        wu = args.warmup if size <= 1024 else 5
        torch.cuda.empty_cache()
        runners = builder(size, dtype, device, direction=args.direction)
        row = {"size": size}
        for k in keys:
            if k == "torch" and (args.no_torch or size > args.torch_max):
                row[k] = float("nan")
                continue
            row[k] = time_ms(runners[k], it, wu)
        rows.append(row)
        cu = row["cueq"]
        parts = []
        for k in keys:
            v = row[k]
            tag = f"{v:.3f}" if v == v else "nan"
            if k.startswith("triangle_ops") and v == v and cu == cu:
                tag += f"({cu / v:.2f}x)"
            parts.append(f"{k}={tag}")
        print(f"  L={size:5d}  " + "  ".join(parts))

    csv_path = os.path.join(out_dir, f"{args.op}.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["size"] + keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"Wrote {csv_path}")

    try:
        make_plot(rows, args.op, args.dtype, out_dir)
    except ImportError:
        print("(matplotlib unavailable — skipping plot)")


if __name__ == "__main__":
    main()
