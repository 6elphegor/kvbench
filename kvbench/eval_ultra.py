"""Ultra-long-context evaluation sweep: retrieval accuracy vs context length
out to 10M tokens, via the streaming forward in longctx.py.

    python -m kvbench.eval_ultra --runs-dir runs/ultra_dropout_b8 \
        --variants hybrid_square --dim 256 --heads 4 --blocks 8

Writes/updates <runs-dir>/accuracy_vs_context_ultra.json after every point, so
an interrupted sweep loses at most one point. Sequences per point scale down
with n (a single n=416666 sequence already contains 416k scored lookups).
"""
import argparse
import json
import os
import time

import torch

from . import config as C

C.apply_ultra()

from .data import gen_batch, value_positions, second_occurrence_flags  # noqa: E402
from .model import Model, VARIANTS  # noqa: E402
from .longctx import predict_stream  # noqa: E402

# 10M tokens / (2 keys-worth of entries * 12 tokens) = 416,666 keys
DEFAULT_SWEEP = [1000, 2000, 4000, 8000, 16000, 32000, 64000,
                 128000, 256000, 416666]


@torch.no_grad()
def point_accuracy(model, n_keys, seqs):
    ne = 2 * n_keys
    valp = value_positions(ne)
    correct = total = 0
    for _ in range(seqs):
        seq = gen_batch(1, ne)
        preds = predict_stream(model, seq[:, :-1])
        pv = preds[:, valp].view(1, ne, C.VLEN)
        tv = seq[:, 1:][:, valp].view(1, ne, C.VLEN)
        sec = second_occurrence_flags(seq, ne)
        exact = (pv == tv).all(-1)
        correct += exact[sec].sum().item()
        total += int(sec.sum().item())
        del seq, preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return correct / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/ultra")
    ap.add_argument("--variants", default="hybrid_square")
    ap.add_argument("--sweep", default=",".join(map(str, DEFAULT_SWEEP)))
    ap.add_argument("--seqs-budget", type=int, default=131072,
                    help="~lookups per point: seqs = clamp(budget//n, 1, 32)")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step (default: ckpt_final.pt)")
    ap.add_argument("--dim", type=int, default=None, help="model dim override")
    ap.add_argument("--heads", type=int, default=None, help="head count override")
    ap.add_argument("--blocks", type=int, default=None,
                    help="local/global pair count override")
    ap.add_argument("--mutant-j", type=int, default=0,
                    help="evaluate on the cluster data variant (j mutant "
                         "neighbors per key; supersedes the mutated default)")
    args = ap.parse_args()
    if args.mutant_j:
        C.MUTANT_J = args.mutant_j
    if args.dim or args.heads or args.blocks:
        C.set_model(args.dim, args.heads, args.blocks)
    sweep = [int(x) for x in args.sweep.split(",")]
    out_path = os.path.join(args.runs_dir, "accuracy_vs_context_ultra.json")
    data = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for name in args.variants.split(","):
        ckdir = os.path.join(args.runs_dir, name, "checkpoints")
        path = (os.path.join(ckdir, f"ckpt_step{args.step}.pt") if args.step
                else os.path.join(ckdir, "ckpt_final.pt"))
        ck = torch.load(path, map_location=C.DEVICE)
        model = Model(VARIANTS[name]).to(C.DEVICE)
        model.load_state_dict(ck["model_state"])
        model.eval()
        curve = data.setdefault(name, {})
        for n in sweep:
            if str(n) in curve:
                print(f"[{name}] n={n}: cached {curve[str(n)]['acc']:.4f}")
                continue
            seqs = max(1, min(32, args.seqs_budget // n))
            t0 = time.time()
            acc, total = point_accuracy(model, n, seqs)
            curve[str(n)] = {"acc": acc, "lookups": total, "seqs": seqs,
                             "tokens": 2 * n * C.ENTRY,
                             "secs": round(time.time() - t0, 1)}
            json.dump(data, open(out_path, "w"), indent=2)
            print(f"[{name}] n={n} ({2*n*C.ENTRY/1e6:.2f}M tokens): "
                  f"acc {acc:.4f} over {total} lookups "
                  f"({seqs} seqs, {time.time()-t0:.0f}s)", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("done:", out_path)


if __name__ == "__main__":
    main()
