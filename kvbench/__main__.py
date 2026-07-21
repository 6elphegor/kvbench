"""Run the full experiment: train all variants, then render the figures.

    python -m kvbench --compile
"""
import argparse
import os

from .plots import fig_generalization, fig_per_index_ce
from .train import train_variant
from .model import VARIANTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    for name in VARIANTS:
        print(f"=== training {name} ===", flush=True)
        n = train_variant(name, os.path.join(args.runs_dir, name), args.compile)
        print(f"=== {name} done ({n} params) ===", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    # fresh checkpoints -> recompute the accuracy cache
    fig_generalization(args.runs_dir, os.path.join(args.out_dir, "fig_generalization.png"), recompute=True)
    fig_per_index_ce(args.runs_dir, os.path.join(args.out_dir, "fig_per_index_ce.png"))


if __name__ == "__main__":
    main()
