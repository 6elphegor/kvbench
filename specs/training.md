# Training

**Model:** standard backbone, `d_model = 128`, 8 attention + 8 FFN layers,
4 query heads / 4 KV heads (head_dim 32), SwiGLU FFN (hidden 512). ~2.1M params.

**Loss:** cross-entropy on the value digits of **second-occurrence** entries only.
First-occurrence values are unpredictable (nothing to retrieve yet) and all
StartKey / key / StartValue positions are structural, so every gradient term is a
solvable lookup.

**Optimizer:** AdamW, lr `3e-4` (flat, 500-step linear warmup), betas `(0.9, 0.999)`,
weight decay `0.1`, eps `1e-8`.

**Batch size:** 64. **Steps:** 5000. **Seed:** 0 (every variant).

**Logging:** every 100 steps — loss and second-occurrence retrieval accuracy
(per-digit). **Checkpoints + per-index eval:** every 1000 steps.
