"""Hyperparameters for the key-value retrieval context-length experiment.

The task: sequences of `KEY <k digits> VAL <v digits>` entries where every key
appears exactly twice with the same value. The model is trained to predict the
value on the *second* occurrence of a key (an in-context lookup) and evaluated on
how well that retrieval survives at context lengths far longer than training.
"""

# ---- vocabulary ----
START_KEY, START_VALUE = 10, 11
VOCAB = 12                      # digits 0-9 + StartKey + StartValue

# ---- dataset ----
KLEN = 3                        # key digits
VLEN = 3                        # value digits (== KLEN)
ENTRY = 2 + KLEN + VLEN         # tokens per entry: SK + key + SV + value = 8
N_TRAIN_KEYS = 8                # unique keys per training sequence
TRAIN_ENTRIES = 2 * N_TRAIN_KEYS  # each key appears twice -> 16 entries
EVAL_ENTRIES = 100              # unique-keys*2 at the default eval length
MUTATED_KEYS = False            # data variant: half the unique keys are a 1-digit
                                # mutation of another key in the same sequence
MUTANT_J = 0                    # data variant: >0 puts every key in a cluster of
                                # 1 base + MUTANT_J single-digit mutants of it
                                # (supersedes MUTATED_KEYS)
MUTANT_MIX = 0.0                # fraction of sequences per batch drawn with
                                # plain distinct keys instead of clusters

# ---- model ----
D = 128                         # hidden dim
H = 4                           # query heads
KV = 4                          # kv heads (== H -> full MHA)
DK = D // H                     # head dim = 32
BLOCKS = 4                      # local/global pairs -> 2*BLOCKS attn layers
FFN_MULT = 4                    # SwiGLU hidden = FFN_MULT * D = 512
WINDOW = 10                     # sliding-window size for hybrid local layers
ROPE_THETA = 10000.0
DROPOUT = 0.0                   # residual dropout on sublayer outputs


def set_model(dim=None, heads=None, blocks=None):
    """Override model geometry (CLI-driven); rebuilds the variant grid."""
    global D, H, KV, DK, BLOCKS
    if dim:
        D = dim
    if heads:
        H = KV = heads
    if blocks:
        BLOCKS = blocks
    DK = D // H
    from . import model
    model.VARIANTS.clear()
    model.VARIANTS.update(model._build_variants())


# ---- ultra-long-context mode ----
ULTRA = False                   # flipped by apply_ultra(); routes attention
                                # through the fused flex path (longctx.py)


def apply_ultra():
    """Switch to the ultra-long-context experiment configuration:
    7-digit keys, mutated-keys data, a dynamic curriculum (train.py promotes
    through rungs 8, 16, ..., 1000, 2000, 4000, 8000 keys as soon as the
    current rung is solved), length-scaled batch, and the O(T)-memory
    attention path."""
    global ULTRA, KLEN, ENTRY, N_TRAIN_KEYS, TRAIN_ENTRIES, MUTATED_KEYS
    global STEPS, CURRICULUM, WARMUP, WINDOW, LR
    ULTRA = True
    LR = 1.5e-4                 # 3e-4 oscillates/NaNs on the 7-digit mutated
                                # task around rung transitions (observed)
    KLEN = 7                    # 10^7 key space; entry = 2 + 7 + 3 = 12 tokens
    ENTRY = 2 + KLEN + VLEN
    N_TRAIN_KEYS = 1000         # 2000 entries = 24000 tokens
    TRAIN_ENTRIES = 2 * N_TRAIN_KEYS
    MUTATED_KEYS = True
    STEPS = 4000
    WARMUP = 200
    CURRICULUM = ()             # ultra uses the dynamic curriculum in train.py
    WINDOW = ENTRY + 2          # keep a full entry visible, as W=10 does for
                                # the 8-token entries of the base benchmark
    # the variant grid captured WINDOW at import; rebuild it in place
    from . import model
    model.VARIANTS.clear()
    model.VARIANTS.update(model._build_variants())


def batch_for(n_keys):
    """Sequences per step: constant lookups-per-step budget at long n."""
    if ULTRA:
        return max(1, min(BATCH, 2048 // n_keys))
    return BATCH

# ---- training ----
BATCH = 64
STEPS = 15000
WARMUP = 500
LR = 3e-4                       # cosine-decayed to ~0 after warmup
# context-length curriculum: (until_step, n_keys) stages, then N_TRAIN_KEYS.
# Starting at n=2 keys lets the content-match circuit form before any positional
# shortcut exists; without it, every full-attention RoPE variant plateaus ~37%.
CURRICULUM = ((750, 2), (1500, 4), (2500, 6))
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.1
EPS = 1e-8
LOG_EVERY = 100
EVAL_EVERY = 1000
EVAL_BATCHES = 12
SEED = 0

DEVICE = "cuda"
