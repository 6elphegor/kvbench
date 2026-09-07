"""Synthetic key-value data, generated on the fly.

A sequence of `n_entries` entries is built from `n_entries // 2` unique keys with
independently random values; each key-value pair appears exactly twice and the
entries are shuffled. The first occurrence of a key is unpredictable; the second
is a deterministic in-context lookup.

With `mutated` (the ultra-long-context data variant), half of the unique keys
are a 1-digit mutation of another key in the same sequence, so the model must
distinguish nearly identical keys -- the collision mode that dominates as the
context grows. All keys in a sequence remain distinct.

With `mutant_j` > 0 (the cluster variant, superseding `mutated`), every key
belongs to a cluster of one base key plus >= j single-digit mutants of that
base: every retrieval faces at least j in-context near-duplicates, not just
the average-case collisions the mutated variant provides.
"""
import torch

from . import config as C


def _digits(x, n):
    """[B, k] integers -> [B, k, n] base-10 digits, most significant first."""
    return torch.stack([(x // 10 ** (n - 1 - i)) % 10 for i in range(n)], dim=-1)


def _sample_distinct(batch, n, space, device):
    """[B, n] distinct integers from [0, space), by resampling duplicates until
    none remain. For sparse draws this beats materializing an argsort over the
    whole space (at KLEN=7 that would be a 10M-column sort per row per step);
    resampling only the duplicated slots converges even when n is a sizable
    fraction of the space (expected duplicates shrink geometrically)."""
    assert n <= space
    keys = torch.randint(0, space, (batch, n), device=device)
    while True:
        srt, order = keys.sort(dim=1)
        dup_sorted = torch.cat([srt.new_zeros(batch, 1, dtype=torch.bool),
                                srt[:, 1:] == srt[:, :-1]], dim=1)
        if not dup_sorted.any():
            return keys
        dup = torch.zeros_like(dup_sorted)
        dup.scatter_(1, order, dup_sorted)          # later members of each run
        keys[dup] = torch.randint(0, space, (int(dup.sum()),), device=device)


def _resample_mutants(base, src, mut, todo):
    """Fill the `todo` slots of `mut` with single-digit mutations of `src`
    until every key in [base | mut] is distinct within its row. Collision
    detection is sort-based -- a pairwise eq matrix would be O(n^2) memory
    and dies at the 416k-key eval scale."""
    B, n_base = base.shape
    n_mut = mut.shape[1]
    device = base.device
    while todo.any():
        pw = 10 ** (C.KLEN - 1 - torch.randint(0, C.KLEN, (B, n_mut), device=device))
        old = (src // pw) % 10
        new = (old + torch.randint(1, 10, (B, n_mut), device=device)) % 10
        mut = torch.where(todo, src + (new - old) * pw, mut)
        srt, order = torch.cat([base, mut], dim=1).sort(dim=1)
        eq = srt[:, 1:] == srt[:, :-1]
        pad = eq.new_zeros(B, 1)
        dup_sorted = torch.cat([eq, pad], 1) | torch.cat([pad, eq], 1)
        dup = torch.zeros_like(dup_sorted)
        dup.scatter_(1, order, dup_sorted)
        todo = dup[:, n_base:]              # only mutant slots get resampled
    return mut


def _mutate_keys(base, n_mut):
    """[B, n_mut] keys, each differing from a random key in `base` in exactly one
    digit, and distinct from every base key and every other mutant in its row."""
    B, n_base = base.shape
    src = base.gather(1, torch.randint(0, n_base, (B, n_mut), device=base.device))
    mut = base.new_zeros(B, n_mut)
    todo = torch.ones(B, n_mut, dtype=torch.bool, device=base.device)
    return _resample_mutants(base, src, mut, todo)


def _cluster_keys(batch, n_unique, j, device):
    """All keys in mutation clusters: `n_unique // (1+j)` random base keys,
    every remaining slot a single-digit mutation of its cluster's base
    (mutant slots are dealt round-robin, so each base gets at least j
    mutants). Every key in a sequence therefore has at least j in-context
    neighbors within Hamming distance 2 -- the base's mutants are at distance
    1 from it and at most 2 from each other -- and all keys stay distinct
    (colliding mutants are resampled, keeping their cluster assignment)."""
    space = 10 ** C.KLEN
    # the collision-resampling loop needs slack to converge: at n_unique
    # close to the key space the mutants would have to tile the space
    # perfectly, which random resampling never finds (it hangs, not errors)
    assert n_unique <= space // 2, (
        f"mutant clusters need n_unique <= half the key space "
        f"({n_unique} > {space // 2}; raise KLEN)")
    # a base supports at most 9*KLEN distinct 1-digit mutants; keep one spare
    # so the resampling loop always has a free value to converge to
    jj = min(j, n_unique - 1, 9 * C.KLEN - 1)
    if jj <= 0:
        return _sample_distinct(batch, n_unique, space, device)
    nc = max(1, n_unique // (1 + jj))
    n_mut = n_unique - nc
    if space > 10 ** 4:
        bases = _sample_distinct(batch, nc, space, device)
    else:
        bases = torch.argsort(torch.rand(batch, space, device=device),
                              dim=1)[:, :nc]
    src = bases[:, torch.arange(n_mut, device=device) % nc]
    mut = bases.new_zeros(batch, n_mut)
    todo = torch.ones(batch, n_mut, dtype=torch.bool, device=device)
    return torch.cat([bases, _resample_mutants(bases, src, mut, todo)], dim=1)


def gen_batch(batch, n_entries, device=C.DEVICE, mutated=None, mutant_j=None):
    """[B, ENTRY*n_entries] tokens: [SK, key(KLEN), SV, value(VLEN)] * n_entries."""
    if mutated is None:
        mutated = C.MUTATED_KEYS
    if mutant_j is None:
        mutant_j = C.MUTANT_J
    n_unique = n_entries // 2
    n_keys = 10 ** C.KLEN
    if mutant_j:              # cluster variant supersedes the mutated flag
        n_plain = int(round(batch * C.MUTANT_MIX))
        parts = []
        if batch - n_plain:
            parts.append(_cluster_keys(batch - n_plain, n_unique,
                                       mutant_j, device))
        if n_plain:           # plain rows: distinct random keys, no clusters
            parts.append(_sample_distinct(n_plain, n_unique, n_keys, device))
        keys = torch.cat(parts, dim=0)
    else:
        n_mut = n_unique // 2 if mutated else 0
        n_base = n_unique - n_mut
        if n_keys > 10 ** 4:  # large key spaces (KLEN >= 5); the argsort path
            keys = _sample_distinct(batch, n_base, n_keys, device)
        else:                 # small spaces keep the original (committed) RNG
            keys = torch.argsort(torch.rand(batch, n_keys, device=device),
                                 dim=1)[:, :n_base]
        if n_mut:
            keys = torch.cat([keys, _mutate_keys(keys, n_mut)], dim=1)
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
    """[B, n_entries] bool: True where an entry is the 2nd appearance of its
    key. Sort-based (stable in position via a composite sort key), so it scales
    to the 833k-entry ultra eval where a pairwise [B,n,n] mask cannot."""
    e = seq.view(seq.shape[0], n_entries, C.ENTRY)
    kdig = e[..., 1:1 + C.KLEN]
    keys = sum(kdig[..., i] * 10 ** (C.KLEN - 1 - i) for i in range(C.KLEN))  # [B,n]
    pos = torch.arange(n_entries, device=seq.device)
    order = (keys * n_entries + pos).argsort(dim=1)     # by key, then position
    srt = keys.gather(1, order)
    rep_sorted = torch.cat([srt.new_zeros(keys.shape[0], 1, dtype=torch.bool),
                            srt[:, 1:] == srt[:, :-1]], dim=1)
    flags = torch.zeros_like(rep_sorted)
    flags.scatter_(1, order, rep_sorted)
    return flags
