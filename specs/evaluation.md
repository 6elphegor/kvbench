# Evaluation

Two metrics, both restricted to **second-occurrence** value positions (the actual
lookups).

## Context-length generalization (headline)

Exact-match retrieval accuracy (all `v` value digits correct) on second
occurrences, measured at a sweep of context lengths
`n ∈ {8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256}` unique keys. Trained
at n=8, so n=256 is 32× the training length. The RoPE/mask buffers are re-precomputed per context
length (`Model.set_max_entries`). See `kvbench/evaluate.py: accuracy_vs_context`.

## Per-index cross-entropy (mechanism)

Mean cross-entropy at each entry index of a fixed 100-entry evaluation sequence
(averaged over the `v` value digits and over sequences). Shows how retrieval
quality varies with position within a long context. Chance is `ln 10 ≈ 2.30` per
digit. Logged to `runs/<variant>/eval.json` during training.

**Batch size:** 8–64 depending on the metric; **eval batches:** 12–16.
