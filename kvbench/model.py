"""Transformer variants for the retrieval benchmark.

Standard backbone (default nn init, learnable LayerNorm without bias, plain
residuals, plain SwiGLU, standard attention scaling). The variants differ only in
their attention layers:

  * RoPE fraction   -- how much of each head's dims are rotary (1.0, 0.5, or 0.0)
  * sliding window  -- local attention window, or None for full attention
  * squared scores  -- softmax(scores^2) instead of softmax(scores)

`hybrid` variants alternate [local RoPE window] and [global NoPE] attention
layers; `rope`/`partial_rope` variants apply the same attention in every layer.
The `hybrid_square_kda` variant keeps the winning layout but replaces the local
RoPE window layers with Kimi Delta Attention (see `KDA`). Each variant has
8 attention + 8 FFN layers.
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


class KDA(torch.nn.Module):
    """Kimi Delta Attention as used in Kimi K3 (K3 tech report sec. 2.1.1;
    introduced in Kimi Linear, arXiv:2510.26692). A gated DeltaNet whose
    forget gate is per-channel: each head keeps a recurrent state S [d_k, d_v]
    that is decayed channel-wise and then written by the delta rule (replace
    the stored value for the current key at rate beta):

        S_t = (I - beta_t k_t k_t^T) Diag(a_t) S_{t-1} + beta_t k_t v_t^T
        o_t = S_t^T q_t

    Following the report: q/k/v pass through a short causal depthwise conv +
    SiLU, q and k are L2-normalized, beta_t = sigmoid per head, and the output
    gets a per-head RMSNorm and a sigmoid gate. The decay logits z_t come from
    a low-rank projection plus a per-channel bias. K3 changes two things vs
    Kimi Linear: the log-decay is lower-bounded by a scaled sigmoid,
    g_t = g_min * sigmoid(exp(A_h) * z_t) with g_min = -5 and per-head A_h
    init 0 (Kimi Linear used unbounded -exp(A)*softplus(z)), and the output
    gate is full-rank instead of low-rank. No positional encoding and no
    T x T mask: position enters only through the conv and the recurrence, so
    the layer needs no length-dependent buffers.
    """
    CONV = 4      # short-conv kernel size
    RANK = 16     # rank of the decay-logit projection
    GMIN = -5.0   # lower bound of the per-step log-decay
    CHUNK = 16    # chunk length of the scan; |GMIN|*CHUNK = 80 must stay
                  # below ln(float32 max) ~ 88 so exp(-g) cannot overflow

    def __init__(self):
        super().__init__()
        hd = C.H * C.DK
        self.norm = torch.nn.LayerNorm(C.D, bias=False)
        self.Wq = torch.nn.Linear(C.D, hd, bias=False)
        self.Wk = torch.nn.Linear(C.D, hd, bias=False)
        self.Wv = torch.nn.Linear(C.D, hd, bias=False)
        self.conv = torch.nn.Conv1d(3 * hd, 3 * hd, self.CONV, groups=3 * hd, bias=False)
        self.Wf1 = torch.nn.Linear(C.D, self.RANK, bias=False)
        self.Wf2 = torch.nn.Linear(self.RANK, hd, bias=False)
        self.Wbeta = torch.nn.Linear(C.D, C.H, bias=False)
        self.Wg = torch.nn.Linear(C.D, hd, bias=False)      # full-rank output gate (K3)
        self.rms_w = torch.nn.Parameter(torch.ones(C.DK))
        self.Wo = torch.nn.Linear(C.D, C.D, bias=False)
        # per-head log-scale A init 0 (K3); bias init as in GDN / Kimi Linear:
        # dt log-uniform in [1e-3, 1e-1] mapped through softplus^-1
        self.A = torch.nn.Parameter(torch.zeros(C.H))
        dt = torch.exp(torch.empty(hd).uniform_(math.log(1e-3), math.log(1e-1)))
        self.b_alpha = torch.nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        B, T, _ = x.shape
        h = self.norm(x)
        qkv = torch.cat([self.Wq(h), self.Wk(h), self.Wv(h)], dim=-1).transpose(1, 2)
        qkv = F.silu(self.conv(F.pad(qkv, (self.CONV - 1, 0)))).transpose(1, 2)
        q, k, v = qkv.view(B, T, 3, C.H, C.DK).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        z = (self.Wf2(self.Wf1(h)) + self.b_alpha).view(B, T, C.H, C.DK)
        glog = self.GMIN * torch.sigmoid(self.A.exp().view(1, 1, C.H, 1) * z)
        o = self._scan(q, k, v, glog, torch.sigmoid(self.Wbeta(h)))
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * self.rms_w
        o = o * torch.sigmoid(self.Wg(h)).view(B, T, C.H, C.DK)
        return self.Wo(o.reshape(B, T, C.D))

    def _scan(self, q, k, v, glog, beta):
        """Chunkwise-parallel gated delta rule (the papers' WY-style algorithm).

        Within a chunk, substituting S_t = Diag(exp(g_t)) Z_t (g = within-chunk
        cumulative log-decay) turns the recurrence into rank-1 updates
        Z_t = Z_{t-1} + kappa_t r_t^T whose coefficients R solve one unit-lower-
        triangular system; outputs and the carried state are then plain matmuls.
        Only S [B,H,DK,DK] crosses chunk boundaries. The decay-scaled products
        exp(g_t - g_s) are computed as (x*exp(g)) @ (y*exp(-g))^T -- K3's
        lower-bounded decay guarantees exp(-g) <= e^{|GMIN|*CHUNK} stays inside
        float32 range (the reason K3 bounds it). `glog` is the per-step
        log-decay (log a_t in (GMIN, 0)).
        """
        B, T, H, Dk = q.shape
        L = self.CHUNK
        if T % L:                                        # zero-pad to a whole chunk:
            zpad = lambda t: torch.cat(                  # k=0 rows write nothing
                [t, t.new_zeros(B, -T % L, *t.shape[2:])], dim=1)
            q, k, v, glog, beta = map(zpad, (q, k, v, glog, beta))
        # [B,H,NC,L,*]: every within-chunk quantity is batched over the chunks
        q, k, v, glog = (t.view(B, -1, L, H, Dk).permute(0, 3, 1, 2, 4)
                         for t in (q, k, v, glog))
        cb = beta.view(B, -1, L, H).permute(0, 3, 1, 2).unsqueeze(-1)
        g = glog.cumsum(3)
        up, down = g.exp(), (-g).exp()
        qg, kg, kig = q * up, k * up, k * down
        A = (kg @ kig.transpose(-1, -2)).tril(-1)        # k_t.k_s e^{g_t-g_s}, t>s
        Aq = (qg @ kig.transpose(-1, -2)).tril()         # q_t.k_s e^{g_t-g_s}, t>=s
        eye = torch.eye(L, device=q.device).expand(A.shape).contiguous()
        Minv = torch.linalg.solve_triangular(eye + cb * A, eye,
                                             upper=False, unitriangular=True)
        R0 = Minv @ (cb * v)                             # R for S = 0 ...
        W = Minv @ (cb * kg)                             # ... and its S-correction
        gL = g[:, :, :, -1:]
        P = (k * (gL - g).exp()).transpose(-1, -2)       # state-write coefficients
        lam = gL.exp().transpose(-1, -2)                 # chunk-total decay
        # sequential part: three small matmuls per chunk
        S = q.new_zeros(B, H, Dk, Dk)
        outs = []
        for c in range(g.shape[2]):
            R = R0[:, :, c] - W[:, :, c] @ S
            outs.append(qg[:, :, c] @ S + Aq[:, :, c] @ R)
            S = lam[:, :, c] * S + P[:, :, c] @ R
        o = torch.stack(outs, dim=2)                     # [B,H,NC,L,DK]
        return o.permute(0, 2, 3, 1, 4).reshape(B, -1, H, Dk)[:, :T]


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
            elif spec[0] == "kda":
                self.layers.append(KDA())
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

def _hybrid_kda(square):
    # the hybrid layout with KDA in place of the local RoPE window layers
    return [s for _ in range(4) for s in
            (("kda",), _FFN, _attn(0.0, None, square), _FFN)]

VARIANTS = {
    "rope":                    _uniform(1.0, False),
    "rope_square":             _uniform(1.0, True),
    "partial_rope":            _uniform(0.5, False),
    "partial_rope_square":     _uniform(0.5, True),
    "hybrid":                  _hybrid(C.WINDOW, False),
    "hybrid_square":           _hybrid(C.WINDOW, True),
    "hybrid_nowindow":         _hybrid(None, False),
    "hybrid_square_nowindow":  _hybrid(None, True),
    "hybrid_square_kda":       _hybrid_kda(True),
}
