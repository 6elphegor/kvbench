# Architecture

Standard decoder-only transformer, shared by all variants:

- token embedding → N blocks → final `LayerNorm` → output projection
- each block: `x += Attn(LayerNorm(x))`, then `x += FFN(LayerNorm(x))`
- **LayerNorm** is learnable but has no bias; no biases anywhere else
- **FFN** is SwiGLU: `Wd(silu(Wg x) * Wu x)`, hidden = 4·d_model
- **Attention** is grouped-query (here 4 q / 4 kv = full MHA), scores `QKᵀ/√d_k`,
  causal mask, softmax
- default PyTorch initialization; plain residuals (no residual scaling)

Dimensions: `d_model=128`, 8 attention + 8 FFN layers, head_dim 32.

## Variant axes

Variants differ **only** in their attention layers:

- **RoPE fraction**: fraction of each head's dims that get rotary position
  encoding: `1.0` (full RoPE), `0.5` (partial), `0.0` (NoPE, position-free).
- **Sliding window**: local attention window `W`, or `None` for full attention.
- **Squared scores**: `softmax(scores²)` instead of `softmax(scores)`.
- **Accumulated scores**: `softmax(cumsum(scores²))` along the key axis: each
  key's squared score also carries the squared scores of all keys before it.
- **KDA**: Kimi Delta Attention as used in Kimi K3 (K3 tech report §2.1.1;
  introduced in Kimi Linear, arXiv:2510.26692) in place of a softmax attention
  layer: a gated DeltaNet whose per-head recurrent state `S [d_k, d_v]` is
  decayed channel-wise then written by the delta rule,
  `S_t = (I − β_t k_t k_tᵀ) Diag(a_t) S_{t−1} + β_t k_t v_tᵀ`, `o_t = S_tᵀ q_t`.
  Short causal conv + SiLU on q/k/v, L2-normalized q/k, per-head sigmoid β,
  per-head RMSNorm + sigmoid output gate. Includes K3's two changes over Kimi
  Linear: the log-decay is lower-bounded, `g_t = g_min·σ(e^{A_h} z_t)` with
  `g_min = −5` (vs unbounded `−e^A softplus(z)`), and the output gate is
  full-rank rather than low-rank. No positional encoding and no mask:
  position enters only through the conv and the recurrence. Computed with the
  chunkwise-parallel scan.

## The nine variants

| name | attention layers |
|---|---|
| `rope`, `rope_square` | full RoPE, full attention, every layer |
| `partial_rope`, `partial_rope_square` | half-RoPE, full attention, every layer |
| `hybrid`, `hybrid_square` | alternate [full-RoPE + window W=10] and [NoPE + full attention] |
| `hybrid_nowindow`, `hybrid_square_nowindow` | alternate [full-RoPE] and [NoPE], both full attention |
| `hybrid_square_kda` | alternate [KDA] and [NoPE + full attention, squared scores] |

The `_square` variants apply `softmax(scores²)` in every attention layer; in
`hybrid_square_kda` only the NoPE layers have softmax scores to square; KDA
replaces the local layers, playing the sliding window's role (a fading local
memory) without any positional encoding.

The hybrid pattern gives half the layers **no** positional encoding so they can do
purely content-based lookup (length-agnostic), while the RoPE layers are confined
to a short local window, which is what enables generalization to long contexts.
