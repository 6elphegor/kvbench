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

- **RoPE fraction** — fraction of each head's dims that get rotary position
  encoding: `1.0` (full RoPE), `0.5` (partial), `0.0` (NoPE, position-free).
- **Sliding window** — local attention window `W`, or `None` for full attention.
- **Squared scores** — `softmax(scores²)` instead of `softmax(scores)`.
- **Accumulated scores** — `softmax(cumsum(scores²))` along the key axis: each
  key's squared score also carries the squared scores of all keys before it.

## The eight variants

| name | attention layers |
|---|---|
| `rope`, `rope_square` | full RoPE, full attention, every layer |
| `partial_rope`, `partial_rope_square` | half-RoPE, full attention, every layer |
| `hybrid`, `hybrid_square` | alternate [full-RoPE + window W=10] and [NoPE + full attention] |
| `hybrid_nowindow`, `hybrid_square_nowindow` | alternate [full-RoPE] and [NoPE], both full attention |

The `_square` variants apply `softmax(scores²)` in every attention layer.

The hybrid pattern gives half the layers **no** positional encoding so they can do
purely content-based lookup (length-agnostic), while the RoPE layers are confined
to a short local window — which is what enables generalization to long contexts.
