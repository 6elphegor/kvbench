"""Evaluation utilities.

Two views of retrieval quality:
  * per_index_ce  -- cross-entropy at each value position of a fixed-length eval
                     sequence (used during training, saved to eval.json).
  * accuracy_vs_context -- exact-match retrieval accuracy on second-occurrence
                     values as the context length (number of unique keys) grows;
                     this is the context-length-generalization metric.
"""
import torch
import torch.nn.functional as F

from . import config as C
from .data import gen_batch, value_positions, second_occurrence_flags


@torch.no_grad()
def per_index_ce(model, batches=C.EVAL_BATCHES, batch=C.BATCH, n_entries=C.EVAL_ENTRIES):
    """Mean cross-entropy per entry index (averaged over the VLEN value digits)."""
    model.eval()
    ce_sum = torch.zeros(n_entries, device=C.DEVICE)
    valp = value_positions(n_entries)
    for _ in range(batches):
        seq = gen_batch(batch, n_entries)
        logits = model(seq[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, C.VOCAB).float(), seq[:, 1:].reshape(-1),
                             reduction="none").view(batch, -1)
        per_entry = ce[:, valp].view(batch, n_entries, C.VLEN).mean(-1)
        ce_sum += per_entry.sum(0)
    model.train()
    return (ce_sum / (batches * batch)).tolist()


@torch.no_grad()
def retrieval_accuracy(model, n_keys, batches=16, batch=8):
    """Exact-match accuracy (all VLEN digits) on second-occurrence values."""
    model.eval()
    n_entries = 2 * n_keys
    valp = value_positions(n_entries)
    correct = total = 0
    for _ in range(batches):
        seq = gen_batch(batch, n_entries)
        pred = model(seq[:, :-1]).argmax(-1)
        pv = pred[:, valp].view(batch, n_entries, C.VLEN)
        tv = seq[:, 1:][:, valp].view(batch, n_entries, C.VLEN)
        exact = (pv == tv).all(-1)
        sec = second_occurrence_flags(seq, n_entries)
        correct += exact[sec].sum().item()
        total += int(sec.sum().item())
    model.train()
    return correct / total


def accuracy_vs_context(model, n_list, batch=4, batches=32):
    """Retrieval accuracy at each context length in `n_list` (unique-key counts).

    Uses a small batch with more iterations so long contexts (n=256 is a 4096-token
    sequence) stay within GPU memory while keeping ~batch*batches seqs per point.
    """
    model.set_max_entries(2 * max(n_list))
    return [retrieval_accuracy(model, n, batches=batches, batch=batch) for n in n_list]
