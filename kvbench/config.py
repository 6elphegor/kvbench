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

# ---- model ----
D = 128                         # hidden dim
H = 4                           # query heads
KV = 4                          # kv heads (== H -> full MHA)
DK = D // H                     # head dim = 32
FFN_MULT = 4                    # SwiGLU hidden = FFN_MULT * D = 512
WINDOW = 10                     # sliding-window size for hybrid local layers
ROPE_THETA = 10000.0

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
