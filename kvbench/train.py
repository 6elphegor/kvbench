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


def train_variant(name, out_dir, use_compile, seed=None, resume=None, start_step=0,
                  rung_acc=0.999, rung_cap=500):
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

    valp_by_ne = {}
    # ultra mode uses a dynamic curriculum: start at n=8 and advance to the
    # next rung as soon as per-digit retrieval accuracy on the training batch
    # reaches rung_acc. Per-step cost grows ~n^2, so rungs above N_TRAIN_KEYS
    # carry a hard step cap (rung_cap; 0 disables), and training ends early
    # once the final rung holds >= rung_acc for 50 consecutive steps.
    rungs = ([2 ** i for i in range(3, 10)] + [C.N_TRAIN_KEYS]
             + [2000, 4000, 8000])
    rung, rung_step, done_streak = 0, 0, 0
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if not resume:
        open(metrics_path, "w").close()
    evals, t0 = {}, time.time()
    autocast = torch.autocast(C.DEVICE.split(":")[0], dtype=torch.bfloat16,
                              enabled=C.ULTRA)

    def save_final(step):
        torch.save({"step": step, "variant": name, "model_state": model.state_dict()},
                   os.path.join(out_dir, "checkpoints", "ckpt_final.pt"))

    for step in range(start_step + 1, C.STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        nk = rungs[rung] if C.ULTRA else n_keys_at(step)
        ne = 2 * nk
        bsz = C.batch_for(nk)
        valp = valp_by_ne.setdefault(ne, value_positions(ne))
        seq = gen_batch(bsz, ne)
        with autocast:
            logits = fwd(seq[:, :-1])
        tgt = seq[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, C.VOCAB).float(), tgt.reshape(-1),
                             reduction="none").view(bsz, -1)
        sec = second_occurrence_flags(seq, ne)
        ce_val = ce[:, valp].view(bsz, ne, C.VLEN)
        loss = ce_val[sec].mean()                       # 2nd-occurrence values only
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if C.ULTRA:
            # squared-logit grads scale with score magnitude; unclipped spikes
            # at rung transitions NaN the weights (observed). Clip, and skip
            # any step whose gradient is already non-finite.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                print(f"[{name}] step {step}: non-finite grad norm, step skipped",
                      flush=True)
                opt.zero_grad(set_to_none=True)
        opt.step()

        acc = None
        if C.ULTRA or step % C.LOG_EVERY == 0 or step == 1:
            with torch.no_grad():
                acc_logits = logits
                if C.ULTRA and C.DROPOUT:
                    # rung promotion demands ~100% accuracy; the training
                    # forward is dropout-corrupted, so judge on a clean pass
                    model.eval()
                    with autocast:
                        acc_logits = fwd(seq[:, :-1])
                    model.train()
                pv = acc_logits.argmax(-1)[:, valp].view(bsz, ne, C.VLEN)
                tv = tgt[:, valp].view(bsz, ne, C.VLEN)
                acc = (pv[sec] == tv[sec]).float().mean().item()
        if C.ULTRA:
            rung_step += 1
            capped = bool(rung_cap) and nk > C.N_TRAIN_KEYS and rung_step >= rung_cap
            if rung < len(rungs) - 1 and (acc >= rung_acc or capped):
                rung, rung_step = rung + 1, 0
                print(f"[{name}] step {step:5d}  n={nk} at {acc:.3f}"
                      f"{' (cap)' if capped else ''} -> "
                      f"promoting to n={rungs[rung]}", flush=True)
                # rungs past n=4000 can run >20s/step; without these saves a
                # stop there would lose everything since the last 1000-step ckpt
                save_final(step)
            elif rung == len(rungs) - 1:
                done_streak = done_streak + 1 if acc >= rung_acc else 0
                if done_streak >= 50 or capped:
                    save_final(step)
                    print(f"[{name}] step {step:5d}  final rung n={nk} "
                          f"{'solved' if done_streak >= 50 else 'capped'} at "
                          f"{acc:.4f}; stopping", flush=True)
                    return n_params
            if nk >= 4000 and step % 100 == 0:
                save_final(step)

        if step % C.LOG_EVERY == 0 or step == 1:
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

    if C.ULTRA:
        save_final(C.STEPS)
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
    ap.add_argument("--ultra", action="store_true",
                    help="ultra-long-context mode: 7-digit mutated keys, "
                         "dynamic curriculum, fused flex attention")
    ap.add_argument("--rung-acc", type=float, default=0.999,
                    help="per-digit accuracy to clear a curriculum rung (ultra)")
    ap.add_argument("--rung-cap", type=int, default=500,
                    help="max steps on rungs above n=1000 (ultra; 0 = uncapped)")
    ap.add_argument("--dim", type=int, default=None, help="override model dim")
    ap.add_argument("--heads", type=int, default=None, help="override head count")
    ap.add_argument("--blocks", type=int, default=None,
                    help="override local/global pair count (2*blocks attn layers)")
    ap.add_argument("--dropout", type=float, default=None,
                    help="residual dropout on sublayer outputs (default 0)")
    ap.add_argument("--mutant-j", type=int, default=0,
                    help="cluster data variant: every key gets this many "
                         "single-digit mutant neighbors (supersedes the "
                         "ultra default of mutated keys)")
    ap.add_argument("--mutant-mix", type=float, default=None,
                    help="fraction of sequences per batch with plain distinct "
                         "keys instead of mutant clusters")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    if args.ultra:
        C.apply_ultra()
    if args.dim or args.heads or args.blocks:
        C.set_model(args.dim, args.heads, args.blocks)
    if args.steps:
        C.STEPS = args.steps
    if args.dropout is not None:
        C.DROPOUT = args.dropout
    if args.mutant_j:
        C.MUTANT_J = args.mutant_j
    if args.mutant_mix is not None:
        C.MUTANT_MIX = args.mutant_mix

    names = args.only.split(",") if args.only else list(VARIANTS)
    if args.exclude:
        names = [n for n in names if n not in args.exclude.split(",")]
    for name in names:
        print(f"=== training {name} (seed {C.SEED if args.seed is None else args.seed}) ===", flush=True)
        n = train_variant(name, os.path.join(args.runs_dir, name), args.compile, args.seed,
                          args.resume, args.start_step, args.rung_acc, args.rung_cap)
        print(f"=== {name} done ({n} params) ===", flush=True)


if __name__ == "__main__":
    main()
