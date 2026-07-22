"""Train every architecture variant and record metrics + checkpoints.

Loss is computed only on the value digits of second-occurrence entries (the
in-context lookups); first-occurrence values are unpredictable and excluded.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from . import config as C
from .data import gen_batch, value_positions, second_occurrence_flags
from .evaluate import per_index_ce
from .model import Model, VARIANTS


def lr_at(step):
    # short linear warmup, then cosine decay to ~0
    if step < C.WARMUP:
        return C.LR * step / C.WARMUP
    frac = (step - C.WARMUP) / (C.STEPS - C.WARMUP)
    return C.LR * 0.5 * (1.0 + math.cos(math.pi * frac))


def n_keys_at(step):
    # context-length curriculum (see config.CURRICULUM)
    for until, nk in C.CURRICULUM:
        if step <= until:
            return nk
    return C.N_TRAIN_KEYS


def train_variant(name, out_dir, use_compile, seed=None, resume=None, start_step=0):
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    torch.manual_seed(C.SEED if seed is None else seed)
    model = Model(VARIANTS[name]).to(C.DEVICE)
    if resume:
        model.load_state_dict(torch.load(resume, map_location=C.DEVICE)["model_state"])
        print(f"resumed weights from {resume} (starting at step {start_step})", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    fwd = torch.compile(model) if use_compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, betas=C.BETAS,
                            weight_decay=C.WEIGHT_DECAY, eps=C.EPS)

    valp_by_ne = {2 * nk: value_positions(2 * nk)
                  for nk in {n for _, n in C.CURRICULUM} | {C.N_TRAIN_KEYS}}
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if not resume:
        open(metrics_path, "w").close()
    evals, t0 = {}, time.time()

    for step in range(start_step + 1, C.STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ne = 2 * n_keys_at(step)
        valp = valp_by_ne[ne]
        seq = gen_batch(C.BATCH, ne)
        logits = fwd(seq[:, :-1])
        tgt = seq[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, C.VOCAB).float(), tgt.reshape(-1),
                             reduction="none").view(C.BATCH, -1)
        sec = second_occurrence_flags(seq, ne)
        ce_val = ce[:, valp].view(C.BATCH, ne, C.VLEN)
        loss = ce_val[sec].mean()                       # 2nd-occurrence values only
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % C.LOG_EVERY == 0 or step == 1:
            with torch.no_grad():
                pv = logits.argmax(-1)[:, valp].view(C.BATCH, ne, C.VLEN)
                tv = tgt[:, valp].view(C.BATCH, ne, C.VLEN)
                acc = (pv[sec] == tv[sec]).float().mean().item()
            rec = {"step": step, "n_keys": ne // 2, "loss": loss.item(), "retrieval_acc": acc}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{name}] step {step:5d}  n_keys {ne // 2}  loss {loss.item():.4f}  "
                  f"retrieval_acc {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        if step % C.EVAL_EVERY == 0:
            evals[step] = per_index_ce(model)
            torch.save({"step": step, "variant": name, "model_state": model.state_dict()},
                       os.path.join(out_dir, "checkpoints", f"ckpt_step{step}.pt"))
            with open(os.path.join(out_dir, "eval.json"), "w") as f:
                json.dump(evals, f)

    return n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated subset of variants")
    ap.add_argument("--exclude", default=None, help="comma-separated variants to skip")
    ap.add_argument("--seed", type=int, default=None, help="override config.SEED for this run")
    ap.add_argument("--resume", default=None, help="checkpoint .pt to continue training from")
    ap.add_argument("--start-step", type=int, default=0, help="step the resumed checkpoint was at")
    ap.add_argument("--steps", type=int, default=None, help="override config.STEPS (total, not additional)")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    if args.steps:
        C.STEPS = args.steps

    names = args.only.split(",") if args.only else list(VARIANTS)
    if args.exclude:
        names = [n for n in names if n not in args.exclude.split(",")]
    for name in names:
        print(f"=== training {name} (seed {C.SEED if args.seed is None else args.seed}) ===", flush=True)
        n = train_variant(name, os.path.join(args.runs_dir, name), args.compile, args.seed,
                          args.resume, args.start_step)
        print(f"=== {name} done ({n} params) ===", flush=True)


if __name__ == "__main__":
    main()
