"""Transformer variants for the retrieval benchmark.

Standard backbone (default nn init, learnable LayerNorm without bias, plain
residuals, plain SwiGLU, standard attention scaling). The variants differ only in
their attention layers:

  * RoPE fraction   -- how much of each head's dims are rotary (1.0, 0.5, or 0.0)
  * sliding window  -- local attention window, or None for full attention
  * squared scores  -- softmax(scores^2) instead of softmax(scores)

`hybrid` variants alternate [local RoPE window] and [global NoPE] attention
layers; `rope`/`partial_rope` variants apply the same attention in every layer.
Each variant has 8 attention + 8 FFN layers.
"""
import math

import torch
import torch.nn.functional as F

from . import config as C


def apply_rope(x, cos, sin):
    # x [B,T,h,rd]; cos/sin [T, rd/2]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    c = cos[: x.shape[1]].unsqueeze(1)
    s = sin[: x.shape[1]].unsqueeze(1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(torch.nn.Module):
    def __init__(self, rope_frac, window, square):
        super().__init__()
        self.rope_dims = int(C.DK * rope_frac)   # 32, 16, or 0
        self.window = window
        self.square = square
        self.norm = torch.nn.LayerNorm(C.D, bias=False)
        self.Wq = torch.nn.Linear(C.D, C.H * C.DK, bias=False)
        self.Wk = torch.nn.Linear(C.D, C.KV * C.DK, bias=False)
        self.Wv = torch.nn.Linear(C.D, C.KV * C.DK, bias=False)
        self.Wo = torch.nn.Linear(C.D, C.D, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        h = self.norm(x)
        q = self.Wq(h).view(B, T, C.H, C.DK)
        k = self.Wk(h).view(B, T, C.KV, C.DK)
        v = self.Wv(h).view(B, T, C.KV, C.DK)

        if self.rope_dims > 0:
            rd = self.rope_dims
            q = torch.cat([apply_rope(q[..., :rd], cos, sin), q[..., rd:]], dim=-1)
            k = torch.cat([apply_rope(k[..., :rd], cos, sin), k[..., rd:]], dim=-1)

        q = q.transpose(1, 2)                                    # [B,H,T,DK]
        k = k.transpose(1, 2).repeat_interleave(C.H // C.KV, dim=1)
        v = v.transpose(1, 2).repeat_interleave(C.H // C.KV, dim=1)

        scores = q @ k.transpose(-1, -2) / math.sqrt(C.DK)       # [B,H,T,T]
        if self.square:
            scores = scores * scores
        scores = scores.masked_fill(~mask[:T, :T], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, T, C.D)
        return self.Wo(out)


class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden = C.FFN_MULT * C.D
        self.norm = torch.nn.LayerNorm(C.D, bias=False)
        self.Wg = torch.nn.Linear(C.D, hidden, bias=False)
        self.Wu = torch.nn.Linear(C.D, hidden, bias=False)
        self.Wd = torch.nn.Linear(hidden, C.D, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return self.Wd(F.silu(self.Wg(h)) * self.Wu(h))


def _rope_tables(rope_dims, max_t, device):
    if rope_dims == 0:
        z = torch.zeros(1, device=device)
        return z, z
    inv = 1.0 / (C.ROPE_THETA ** (torch.arange(0, rope_dims, 2, device=device).float() / rope_dims))
    freqs = torch.outer(torch.arange(max_t, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def _causal_mask(window, max_t, device):
    i = torch.arange(max_t, device=device)
    mask = i[:, None] >= i[None, :]
    if window is not None:
        mask = mask & (i[:, None] - i[None, :] < window)
    return mask


class Model(torch.nn.Module):
    def __init__(self, sublayers, max_entries=C.EVAL_ENTRIES):
        super().__init__()
        self.E = torch.nn.Embedding(C.VOCAB, C.D)
        self.norm_f = torch.nn.LayerNorm(C.D, bias=False)
        self.W_out = torch.nn.Linear(C.D, C.VOCAB, bias=False)
        self.layers = torch.nn.ModuleList()
        max_t = C.ENTRY * max_entries
        for spec in sublayers:
            if spec[0] == "attn":
                _, rope_frac, window, square = spec
                layer = Attention(rope_frac, window, square)
                cos, sin = _rope_tables(layer.rope_dims, max_t, C.DEVICE)
                layer.register_buffer("cos", cos, persistent=False)
                layer.register_buffer("sin", sin, persistent=False)
                layer.register_buffer("mask", _causal_mask(window, max_t, C.DEVICE), persistent=False)
                self.layers.append(layer)
            else:
                self.layers.append(FFN())

    def set_max_entries(self, max_entries):
        """Re-precompute rope/mask buffers so eval can exceed the training length."""
        max_t = C.ENTRY * max_entries
        for layer in self.layers:
            if isinstance(layer, Attention):
                cos, sin = _rope_tables(layer.rope_dims, max_t, C.DEVICE)
                layer.cos, layer.sin = cos, sin
                layer.mask = _causal_mask(layer.window, max_t, C.DEVICE)

    def forward(self, tokens):
        h = self.E(tokens)
        for layer in self.layers:
            if isinstance(layer, Attention):
                h = h + layer(h, layer.cos, layer.sin, layer.mask)
            else:
                h = h + layer(h)
        return self.W_out(self.norm_f(h))


# ---- variant grid ----
def _attn(rope_frac, window, square):
    return ("attn", rope_frac, window, square)

_FFN = ("ffn",)

def _uniform(rope_frac, square):
    return [s for _ in range(8) for s in (_attn(rope_frac, None, square), _FFN)]

def _hybrid(window, square):
    return [s for _ in range(4) for s in
            (_attn(1.0, window, square), _FFN, _attn(0.0, None, square), _FFN)]

VARIANTS = {
    "rope":                    _uniform(1.0, False),
    "rope_square":             _uniform(1.0, True),
    "partial_rope":            _uniform(0.5, False),
    "partial_rope_square":     _uniform(0.5, True),
    "hybrid":                  _hybrid(C.WINDOW, False),
    "hybrid_square":           _hybrid(C.WINDOW, True),
    "hybrid_nowindow":         _hybrid(None, False),
    "hybrid_square_nowindow":  _hybrid(None, True),
}
