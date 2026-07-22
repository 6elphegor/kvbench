"""Publication-grade figures. Styled to a validated colorblind-safe palette;
only the two hybrid+window winners are colored, the six baselines form a muted
gray band.

    python -m kvbench.plots --runs-dir runs --out-dir figures
"""
import argparse
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter

from . import config as C

# palette
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE, AQUA = "#e1e0d9", "#c3c2b7", "#2a78d6", "#1baf7a"
WINNERS = {"hybrid_square": (BLUE, "hybrid · square + window"),
           "hybrid": (AQUA, "hybrid · window")}
BASELINES = ["rope", "rope_square", "partial_rope", "partial_rope_square",
             "hybrid_nowindow", "hybrid_square_nowindow"]
N_SWEEP = [8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, labelcolor=INK2)
    ax.set_axisbelow(True)


def _titles(fig, main, sub, foot):
    fig.text(0.062, 0.955, main, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.062, 0.888, sub, ha="left", va="top", fontsize=10, color=INK2)
    fig.text(0.062, 0.015, foot, ha="left", va="bottom", fontsize=8.5, color=MUTED)


def _load(runs_dir, name):
    import torch
    from .model import Model, VARIANTS
    ck = torch.load(os.path.join(runs_dir, name, "checkpoints", f"ckpt_step{C.STEPS}.pt"),
                    map_location=C.DEVICE)
    m = Model(VARIANTS[name]).to(C.DEVICE)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def accuracy_curves(runs_dir, recompute=False):
    """Retrieval accuracy vs context length for every variant, cached to
    runs/accuracy_vs_context.json so re-plotting doesn't re-run the eval.
    The cache is invalidated automatically when N_SWEEP changes."""
    names = list(WINNERS) + BASELINES
    cache = os.path.join(runs_dir, "accuracy_vs_context.json")
    if not recompute and os.path.exists(cache):
        data = json.load(open(cache))
        if data.get("n_sweep") == N_SWEEP and all(n in data.get("curves", {}) for n in names):
            print("using cached curves:", cache)
            return data["curves"]
    print("computing accuracy vs context (this is the slow part)...")
    from .evaluate import accuracy_vs_context
    curves = {name: accuracy_vs_context(_load(runs_dir, name), N_SWEEP) for name in names}
    json.dump({"n_sweep": N_SWEEP, "curves": curves}, open(cache, "w"), indent=2)
    print("cached curves:", cache)
    return curves


def _pct(x):
    return f"{round(100 * x, 1):g}%"


def fig_generalization(runs_dir, out_path, recompute=False):
    curves = accuracy_curves(runs_dir, recompute)
    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.135)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.5)
    for name in BASELINES:
        ax.plot(N_SWEEP, curves[name], color=MUTED, alpha=0.5, lw=1.3, zorder=2)
    for name, (col, lab) in WINNERS.items():
        ax.plot(N_SWEEP, curves[name], color=col, lw=2.7, marker="o", ms=6.5,
                mfc=col, mec=SURF, mew=1.4, zorder=5)
    ax.axvline(8, color=MUTED, ls=(0, (4, 3)), lw=1.1, zorder=1)
    ax.text(8, 1.045, "trained here (n=8)", color=MUTED, fontsize=8.5, ha="center", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_SWEEP)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(7.3, 300)
    ax.set_ylim(-0.03, 1.06)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.set_xlabel("context length — unique keys  n   (32× the training length at n=256)")
    ax.set_ylabel("value retrieval accuracy  (exact match, all 3 digits)")
    n_max = N_SWEEP[-1]
    sq, pl = min(curves["hybrid_square"]), curves["hybrid"][-1]
    ax.annotate(f"squared hybrid holds ~{_pct(sq)} to {n_max // C.N_TRAIN_KEYS}×;\n"
                f"plain hybrid softens to ~{_pct(pl)} at n={n_max}",
                xy=(250, 0.935), xytext=(112, 0.76),
                fontsize=9.5, color=INK2, ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.text(20, 0.10, "standard RoPE, partial-RoPE, and\nwindow-free hybrids collapse to 0",
            color=MUTED, fontsize=9.5, ha="left")
    handles = [Line2D([], [], color=c, lw=2.7, marker="o", ms=6.5, mfc=c, mec=SURF, mew=1.4, label=l)
               for _, (c, l) in WINNERS.items()]
    handles.append(Line2D([], [], color=MUTED, alpha=0.6, lw=1.3, label="6 baseline variants"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.02, 0.52),
              frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.8)
    _titles(fig, "Only local-window + global-NoPE hybrids generalize to longer contexts",
            "Key–value retrieval: accuracy on the 2nd occurrence of each key, vs context length. "
            "Trained at n=8; evaluated to n=256.",
            "8 variants · d=128 · 8 blocks · 15k steps · exact-match over 3 value digits · 128 seqs/point")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("saved", out_path)


def fig_per_index_ce(runs_dir, out_path):
    def per_index(name):
        ev = json.load(open(os.path.join(runs_dir, name, "eval.json")))
        return ev[str(max(int(s) for s in ev))]
    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.135)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    x = list(range(1, C.EVAL_ENTRIES + 1))
    for name in BASELINES:
        ax.plot(x, per_index(name), color=MUTED, alpha=0.5, lw=1.3, zorder=2)
    for name, (col, lab) in WINNERS.items():
        ax.plot(x, per_index(name), color=col, lw=2.4, zorder=5)
    ax.set_xlim(1, 100)
    ax.set_ylim(-0.3, 9.2)
    ax.set_xlabel("value position in the sequence  (entry index, 1 → 100)")
    ax.set_ylabel("cross-entropy  (lower = better;  ln10 ≈ 2.30 = chance)")
    ax.axhline(2.3026, color=MUTED, ls=(0, (2, 3)), lw=1, zorder=1)
    ax.text(99, 2.45, "chance", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    ax.annotate("hybrids drive CE → 0 as keys\nrepeat deeper into the context", xy=(78, 1.55),
                xytext=(12, 0.7), fontsize=9.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.text(20, 8.75, "baselines stay high everywhere — no retrieval at this context length",
            color=MUTED, fontsize=9.5, ha="left", va="top")
    handles = [Line2D([], [], color=c, lw=2.4, label=l) for _, (c, l) in WINNERS.items()]
    handles.append(Line2D([], [], color=MUTED, alpha=0.6, lw=1.3, label="6 baseline variants"))
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(0.98, 0.62),
              frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.8)
    _titles(fig, "Inside a 100-key context, only hybrids resolve the values",
            "Cross-entropy at each value position of a single 100-entry evaluation sequence "
            "(mean over sequences).",
            "same 8 variants · 100-entry eval · early positions are unseen keys (nothing to retrieve yet)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--recompute", action="store_true",
                    help="force re-running the accuracy-vs-context eval (ignore the cache)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig_generalization(args.runs_dir, os.path.join(args.out_dir, "fig_generalization.png"), args.recompute)
    fig_per_index_ce(args.runs_dir, os.path.join(args.out_dir, "fig_per_index_ce.png"))


if __name__ == "__main__":
    main()
