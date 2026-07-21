"""Synthetic key-value data, generated on the fly.

A sequence of `n_entries` entries is built from `n_entries // 2` unique keys with
independently random values; each key-value pair appears exactly twice and the
entries are shuffled. The first occurrence of a key is unpredictable; the second
is a deterministic in-context lookup.
"""
import torch

from . import config as C


def _digits(x, n):
    """[B, k] integers -> [B, k, n] base-10 digits, most significant first."""
    return torch.stack([(x // 10 ** (n - 1 - i)) % 10 for i in range(n)], dim=-1)


def gen_batch(batch, n_entries, device=C.DEVICE):
    """[B, ENTRY*n_entries] tokens: [SK, key(KLEN), SV, value(VLEN)] * n_entries."""
    n_unique = n_entries // 2
    n_keys = 10 ** C.KLEN
    keys = torch.argsort(torch.rand(batch, n_keys, device=device), dim=1)[:, :n_unique]
    kdig = _digits(keys, C.KLEN)                                        # [B,nu,KLEN]
    vals = torch.randint(0, 10, (batch, n_unique, C.VLEN), device=device)
    sk = torch.full((batch, n_unique, 1), C.START_KEY, dtype=torch.long, device=device)
    sv = torch.full((batch, n_unique, 1), C.START_VALUE, dtype=torch.long, device=device)
    entries = torch.cat([sk, kdig, sv, vals], dim=-1).repeat(1, 2, 1)   # [B,2nu,ENTRY]
    order = torch.argsort(torch.rand(batch, n_entries, device=device), dim=1)
    entries = torch.gather(entries, 1, order.unsqueeze(-1).expand(-1, -1, C.ENTRY))
    return entries.reshape(batch, n_entries * C.ENTRY)


def value_positions(n_entries, device=C.DEVICE):
    """Positions in the [B, T-1] prediction array that predict value digits."""
    j = torch.arange(n_entries, device=device).unsqueeze(1)
    off = (1 + C.KLEN) + torch.arange(C.VLEN, device=device)
    return (j * C.ENTRY + off).reshape(-1)                             # [n_entries*VLEN]


def second_occurrence_flags(seq, n_entries):
    """[B, n_entries] bool: True where an entry is the 2nd appearance of its key."""
    e = seq.view(seq.shape[0], n_entries, C.ENTRY)
    kdig = e[..., 1:1 + C.KLEN]
    keys = sum(kdig[..., i] * 10 ** (C.KLEN - 1 - i) for i in range(C.KLEN))  # [B,n]
    eq = keys.unsqueeze(2) == keys.unsqueeze(1)                          # [B,i,j]
    earlier = torch.arange(n_entries, device=seq.device)
    earlier = earlier.unsqueeze(1) < earlier.unsqueeze(0)               # [i,j], i<j
    return (eq & earlier.unsqueeze(0)).any(dim=1)                        # [B,n]
