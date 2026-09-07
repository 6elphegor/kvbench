"""Long-context execution paths: fused FlexAttention for training and a
streaming evaluator for contexts far beyond dense-mask reach (10M+ tokens).

The dense path in `model.Attention.forward` materializes [B,H,T,T] scores and a
[T,T] mask; its ceiling on an 80GB GPU is T ~ 50-60k. This module provides:

  * `flex_forward`   -- training/eval forward for an Attention layer using a
                        fused online-softmax kernel (squared scores as a
                        score_mod, sliding window as a block mask) with O(T)
                        memory.
  * `predict_stream` -- no-grad chunked forward of a whole model returning
                        argmax predictions, for 10M-token evaluation, with a
                        coarse block mask (the 128-granularity mask at T=10M
                        would need ~24GB of block indices; 1024-granularity
                        needs ~0.4GB).

Both are used only under `config.ULTRA` (see `config.apply_ultra`).

Hard-won kernel notes:
  * flex_attention must be entered through torch.compile(fullgraph=True,
    dynamic=False); the eager wrapper may materialize score matrices. (Eager
    is used only when CUDA is absent, so the code paths can be tested on CPU.)
  * One compile per distinct T -- raise the dynamo cache limit or shapes
    thrash the cache.
  * ROWS_GUARANTEED_SAFE / BLOCKS_ARE_CONTIGUOUS are promises about
    causal-from-zero attention only. A sliding window breaks both: rows give
    NaN, contiguity silently wrong output. Only set them on global layers.
  * Do not pin kernel tile sizes; the autotuner must pick (128 overflows
    shared memory on sm_86).
"""
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from . import config as C
from .model import apply_rope

torch._dynamo.config.cache_size_limit = 64

_fused_flex = (torch.compile(flex_attention, fullgraph=True, dynamic=False)
               if torch.cuda.is_available() else flex_attention)


def _square_mod(score, b, h, q_idx, kv_idx):
    return score * score


def _kernel_opts(window):
    """Safety promises hold only for causal-from-zero attention; a sliding
    window violates them (NaN rows / silently wrong output). No BACKEND key:
    flex is triton-backed on CUDA, and old nightlies template unknown keys
    straight into the kernel source where the value is an undefined name."""
    if window is None and torch.cuda.is_available():
        return {'ROWS_GUARANTEED_SAFE': True, 'BLOCKS_ARE_CONTIGUOUS': True}
    return {}


_MASK_CACHE = {}


def _block_mask(T, device, window, block=128):
    key = (T, device.type, device.index, window, block)
    if key not in _MASK_CACHE:
        if window is not None:
            def mod(b, h, q, k):
                return (k <= q) & (q - k < window)
        else:
            def mod(b, h, q, k):
                return k <= q
        _MASK_CACHE[key] = create_block_mask(
            mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device,
            BLOCK_SIZE=(block, block), _compile=torch.cuda.is_available())
    return _MASK_CACHE[key]


_ROPE_CACHE = {}


def _rope(rope_dims, T, device):
    """Grow-once rope tables (kvbench half-split convention)."""
    key = (rope_dims, device.type, device.index)
    if key not in _ROPE_CACHE or _ROPE_CACHE[key][0].shape[0] < T:
        inv = 1.0 / (C.ROPE_THETA ** (
            torch.arange(0, rope_dims, 2, device=device).float() / rope_dims))
        freqs = torch.outer(torch.arange(T, device=device).float(), inv)
        _ROPE_CACHE[key] = (freqs.cos(), freqs.sin())
    cos, sin = _ROPE_CACHE[key]
    return cos[:T], sin[:T]


def _qkv(layer, x):
    """LayerNorm + projections + rope, -> [B,H,T,DK] each."""
    B, T, _ = x.shape
    h = layer.norm(x)
    q = layer.Wq(h).view(B, T, C.H, C.DK)
    k = layer.Wk(h).view(B, T, C.KV, C.DK)
    v = layer.Wv(h).view(B, T, C.KV, C.DK)
    if layer.rope_dims > 0:
        cos, sin = _rope(layer.rope_dims, T, x.device)
        rd = layer.rope_dims
        # fp32 rope tables promote q/k out of bf16 under autocast; flex
        # validation (rightly) rejects mixed-dtype q/k/v -- cast back
        q = torch.cat([apply_rope(q[..., :rd], cos, sin), q[..., rd:]], dim=-1).to(v.dtype)
        k = torch.cat([apply_rope(k[..., :rd], cos, sin), k[..., rd:]], dim=-1).to(v.dtype)
    return (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))


def flex_forward(layer, x):
    """Drop-in replacement for the dense Attention.forward body."""
    B, T, _ = x.shape
    q, k, v = _qkv(layer, x)
    out = _fused_flex(
        q, k, v,
        score_mod=_square_mod if layer.square else None,
        block_mask=_block_mask(T, x.device, layer.window),
        scale=C.DK ** -0.5,
        kernel_options=_kernel_opts(layer.window))
    out = out.transpose(1, 2).reshape(B, T, C.D)
    return layer.drop(layer.Wo(out))


# ------------------------------------------------------- streaming evaluation

def _cast_weights(model, dtype):
    """Cached dtype copies of every Linear weight (keyed on the module)."""
    cache = getattr(model, '_wcache', {})
    if cache.get('dtype') != dtype:
        cache = {'dtype': dtype}
        cache.update({m: m.weight.to(dtype) for m in model.modules()
                      if isinstance(m, torch.nn.Linear)})
        model._wcache = cache
    return cache


def _lin(w16, m, x):
    return x @ w16[m].t()


@torch.no_grad()
def _attn_stream(layer, h, w16, q_chunk, dtype=torch.bfloat16):
    """One attention sublayer over the full [1,T,D] fp32 residual, chunked:
    fused flex with a coarse (1024) block mask; projections and matmuls in
    bf16, softmax/accumulators in fp32."""
    T = h.shape[1]
    dev = h.device
    H, KV, DK = C.H, C.KV, C.DK
    # LN + projections, chunked over rows, bf16 outputs
    q = torch.empty(1, T, H * DK, dtype=dtype, device=dev)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    for s in range(0, T, q_chunk * 4):
        e = min(s + q_chunk * 4, T)
        ln = layer.norm(h[:, s:e]).to(dtype)
        q[:, s:e] = _lin(w16, layer.Wq, ln)
        k[:, s:e] = _lin(w16, layer.Wk, ln)
        v[:, s:e] = _lin(w16, layer.Wv, ln)
    q = q.view(1, T, H, DK)
    k = k.view(1, T, KV, DK)
    v = v.view(1, T, KV, DK)
    if layer.rope_dims > 0:
        rd = layer.rope_dims
        for s in range(0, T, 1 << 21):
            e = min(s + (1 << 21), T)
            inv = 1.0 / (C.ROPE_THETA ** (
                torch.arange(0, rd, 2, device=dev).float() / rd))
            fr = torch.outer(torch.arange(s, e, device=dev).float(), inv)
            cs, sn = fr.cos().unsqueeze(1), fr.sin().unsqueeze(1)
            for t in (q, k):
                # .float() on fp32 aliases t -- compute both halves first
                x1 = t[0, s:e, :, :rd // 2].float()
                x2 = t[0, s:e, :, rd // 2:rd].float()
                r1 = (x1 * cs - x2 * sn).to(dtype)
                r2 = (x1 * sn + x2 * cs).to(dtype)
                t[0, s:e, :, :rd // 2] = r1
                t[0, s:e, :, rd // 2:rd] = r2
    q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))  # [1,H,T,DK]

    # triton flex templates only do 32-bit indexing: tensors over 2^31
    # elements (e.g. [1,4,10M,64]) must be split. Heads are independent,
    # so per-head-group calls are exact.
    groups = 1
    while H // groups > 1 and T * (H // groups) * DK >= 2 ** 31:
        groups *= 2
    bm = _block_mask(T, dev, layer.window, block=1024)
    hg = H // groups
    out = torch.cat([
        _fused_flex(
            q[:, g * hg:(g + 1) * hg], k[:, g * hg:(g + 1) * hg],
            v[:, g * hg:(g + 1) * hg],
            score_mod=_square_mod if layer.square else None,
            block_mask=bm, scale=DK ** -0.5,
            kernel_options=_kernel_opts(layer.window))
        for g in range(groups)], dim=1)
    out = out.transpose(1, 2).reshape(1, T, C.D)
    res = torch.empty(1, T, C.D, dtype=torch.float32, device=dev)
    for s in range(0, T, q_chunk * 4):
        e = min(s + q_chunk * 4, T)
        res[:, s:e] = _lin(w16, layer.Wo, out[:, s:e]).float()
    return res


@torch.no_grad()
def predict_stream(model, tokens, q_chunk=8192, dtype=torch.bfloat16):
    """Chunked full-model forward -> argmax next-token predictions [1, T].

    fp32 residual stream ([1,T,D] = 40MB/M tokens at D=128), bf16 matmuls,
    O(T) attention memory. At T=10M peaks around 25GB.
    """
    from .model import Attention
    dev = tokens.device
    T = tokens.shape[1]
    w16 = _cast_weights(model, dtype)
    h = model.E(tokens).float()
    for layer in model.layers:
        if isinstance(layer, Attention):
            h += _attn_stream(layer, h, w16, q_chunk, dtype)
        else:                            # FFN, chunked over rows
            for s in range(0, T, q_chunk * 4):
                e = min(s + q_chunk * 4, T)
                ln = layer.norm(h[:, s:e]).to(dtype)
                mix = F.silu(_lin(w16, layer.Wg, ln)) * _lin(w16, layer.Wu, ln)
                h[:, s:e] += _lin(w16, layer.Wd, mix).float()
    preds = torch.empty(1, T, dtype=torch.long, device=dev)
    for s in range(0, T, q_chunk * 4):
        e = min(s + q_chunk * 4, T)
        ln = model.norm_f(h[:, s:e]).to(dtype)
        preds[:, s:e] = _lin(w16, model.W_out, ln).argmax(-1)
    return preds
