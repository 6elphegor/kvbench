"""Variance of attention-output components at random init vs context length.

Directly simulates a randomly initialised attention head with the benchmark's
dimensions (D=128 hidden, DK=32 head dim, from kvbench/config.py), instead of
the i.i.d.-Gaussian proxy softmax(z1) . z2 in softmax_weighted_sum_cubed.py:

  * hidden states x_i ~ N(0, I_D) for i = 1..n, passed through LayerNorm
  * W_q, W_k, W_v drawn fresh per sample (see --init)
  * q = W_q h_n (the last position attends causally over all n keys, itself
    included); k_i = W_k h_i; v_i = W_v h_i
  * scores s_i = q . k_i / sqrt(DK), optionally raised to a power
  * o = sum_i softmax(s)_i v_i

The plotted quantity is the variance of the DK components of o pooled over
samples. Nothing is assumed about the independence of q, k and v or about the
distribution of the scores; every dependence a real init has is present.
RoPE is omitted: at init a rotation of Gaussian q/k leaves the score
distribution unchanged.

--init unit   W entries ~ N(0, 1/D), so q, k, v components have unit variance
              (the assumption stated in the README).  Default.
--init torch  PyTorch's default nn.Linear init, U(-1/sqrt(D), 1/sqrt(D)),
              i.e. what the trained models actually start from.  q, k, v then
              have variance 1/3 and the scores variance 1/9.

    python attention_output_variance.py            # -> figures/attention_output_variance.png
    python attention_output_variance.py --init torch --out figures/other.png
"""
import argparse
import math
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from kvbench import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
N_GRID = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
POWERS = {"none": 1, "squared": 2, "cubed": 3}

# palette shared with kvbench/plots.py (validated colorblind-safe)
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE, AQUA, AMBER = "#e1e0d9", "#c3c2b7", "#2a78d6", "#1baf7a", "#d97706"
STYLE = {"squared": (BLUE, "o", "Squared"),
         "cubed": (AMBER, "d", "Cubed"),
         "none": (AQUA, "s", "None (raw)")}


def sample_weights(batch, init, gen):
    if init == "unit":
        return torch.randn(batch, 3, C.DK, C.D, generator=gen) / math.sqrt(C.D)
    bound = 1.0 / math.sqrt(C.D)                    # nn.Linear default
    return (torch.rand(batch, 3, C.DK, C.D, generator=gen) * 2 - 1) * bound


def output_components(n, power, init, samples, batch, gen):
    """Returns a 1-D tensor of samples*DK output components."""
    outs = []
    for start in range(0, samples, batch):
        b = min(batch, samples - start)
        x = torch.randn(b, n, C.D, generator=gen)
        h = F.layer_norm(x, (C.D,))
        W = sample_weights(b, init, gen)             # [b, 3, DK, D]
        q = torch.einsum("bd,bkd->bk", h[:, -1], W[:, 0])           # [b, DK]
        k = torch.einsum("bnd,bkd->bnk", h, W[:, 1])                # [b, n, DK]
        v = torch.einsum("bnd,bkd->bnk", h, W[:, 2])                # [b, n, DK]
        s = torch.einsum("bk,bnk->bn", q, k) / math.sqrt(C.DK)      # [b, n]
        s = s ** power
        alpha = torch.softmax(s, dim=-1)
        o = torch.einsum("bn,bnk->bk", alpha, v)                    # [b, DK]
        outs.append(o.reshape(-1))
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["unit", "torch"], default="unit")
    ap.add_argument("--samples", type=int, default=2048, help="heads sampled per n")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "attention_output_variance.png"))
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(args.seed)

    var = {t: [] for t in POWERS}
    print(f"init={args.init}  D={C.D}  DK={C.DK}  samples/n={args.samples}")
    print(f"{'n':>6} " + " ".join(f"{t:>8}" for t in POWERS))
    for n in N_GRID:
        for t, p in POWERS.items():
            o = output_components(n, p, args.init, args.samples, args.batch, gen)
            var[t].append(o.var().item())
        print(f"{n:>6} " + " ".join(f"{var[t][-1]:8.4f}" for t in POWERS))

    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    for t in ["squared", "cubed", "none"]:
        color, marker, label = STYLE[t]
        ax.plot(N_GRID, var[t], marker=marker, color=color, label=label,
                markersize=7, linewidth=2, markeredgecolor=SURF, markeredgewidth=1)
    ax.axhline(1, color=INK2, linestyle="--", linewidth=1, alpha=0.6, label="Var = 1")
    ax.plot(N_GRID, [1 / n for n in N_GRID], color=INK2, linestyle=":", linewidth=1.5,
            alpha=0.8, label="1/n")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Context length (n)", fontsize=12)
    ax.set_ylabel("Variance of attention-output components", fontsize=12)
    ax.set_title(f"Attention output variance at random init  (D={C.D}, head dim {C.DK}, "
                 f"{args.init} init)", fontsize=13)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
