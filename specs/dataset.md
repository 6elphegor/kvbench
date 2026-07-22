# Dataset

**Vocabulary:** digits `0–9`, `StartKey`, `StartValue` (`V = 12`).

**Parameters:** key length `k = 3`, value length `v = 3`.

**Entry:** `StartKey <k digits> StartValue <v digits>` — 8 tokens.

**Sequence:** a sequence of `n_entries` entries is built from `n_entries / 2`
distinct keys, each with an independently random value. Every key-value pair
appears **exactly twice**, and the entries are shuffled. The first occurrence of a
key carries an unpredictable value; the second occurrence repeats the same value,
so it can be predicted by in-context lookup.

Data is generated synthetically on the fly (`kvbench/data.py: gen_batch`).

**Example** (`k=3, v=3`, one pair shown twice):
```
StartKey 6 6 6 StartValue 7 1 4  ...  StartKey 6 6 6 StartValue 7 1 4
```

**Training:** `n = 8` unique keys → 16 entries (128 tokens).
**Evaluation:** up to `n = 256` unique keys → 512 entries (4096 tokens).
