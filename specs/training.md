# Training

**Model:** standard backbone, `d_model = 128`, 8 attention + 8 FFN layers,
4 query heads / 4 KV heads (head_dim 32), SwiGLU FFN (hidden 512). ~2.1M params.

**Loss:** cross-entropy on the value digits of **second-occurrence** entries only.
First-occurrence values are unpredictable (nothing to retrieve yet) and all
StartKey / key / StartValue positions are structural, so every gradient term is a
solvable lookup.

**Optimizer:** AdamW, lr `3e-4` (500-step linear warmup, then cosine decay to ~0),
betas `(0.9, 0.999)`, weight decay `0.1`, eps `1e-8`.

**Batch size:** 64. **Steps:** 15000. **Seed:** 0 for every variant except
`hybrid`, which uses seed 1 (with seed 0 the plain hybrid plateaus at ~78%
retrieval on a partial positional shortcut; seed 1 reaches ≈100%).

**Context-length curriculum:** `n = 2` keys until step 750, `n = 4` until 1500,
`n = 6` until 2500, then `n = 8` for the rest. Without the curriculum, every
variant whose RoPE layers see full attention (rope, partial_rope, the no-window
hybrids) falls into a positional-shortcut local optimum and plateaus at ~37%
exact-match retrieval *at the training length*; starting at `n = 2` — where no
positional shortcut exists — lets the content-based match circuit form first, and
all variants then reach ≈100% at `n = 8` (rope, the slowest, needs the full 15k
steps: 99.9%). The windowed hybrids are unaffected — they learn the task with or
without the curriculum.

**Logging:** every 100 steps — loss and second-occurrence retrieval accuracy
(per-digit). **Checkpoints + per-index eval:** every 1000 steps.
