# Step 11 — Does Tomer's canonicalization help our linking? (cross-evaluation)

Evaluates **Tomer's NLP_ADV pipeline** (segmentation + canonicalization, Gemma-4-31B)
against **our** Step-10 linking geometry. Tomer's research question — *does rewriting
each segment into neutral 3rd-person Hebrew (canonicalization) improve embedding
quality vs. raw text?* — bears directly on Step 10: the bureaucratic "budget register"
that canonicalization removes is exactly the confound Step 10's whitening tried (and
failed, on honesty grounds) to strip geometrically.

## Data
Tomer has run **19 protocols** (early 2022-11/12) fully through segmentation +
canonicalization: **656 segments** (443 substantive, 213 procedural). We reconstruct
the **raw** segment text from `utterance_ids`, pair it with his **canonical** rewrite,
and use his LLM `subject` field as a pseudo-topic label. Both variants are embedded
with the same `multilingual-e5-large` we use in Step 10.

## Results

**1. Intrinsic topic separation (same/different-subject AUC — higher is better):**

| variant | all pairs | substantive only |
|---|---|---|
| raw | 0.901 | 0.947 |
| **canonical** | **0.956** | **0.986** |

Canonicalization **clearly sharpens topic structure** in the embedding space
(substantive AUC 0.947 → 0.986) — it supports Tomer's hypothesis and is the same
intrinsic metric Steps 3/4 used.

**2. Downstream streaming linking (our Step-10 method, recurring subjects as gold):**

| variant | best F1 | precision | recall |
|---|---|---|---|
| raw | 0.114 | 0.073 | 0.261 |
| canonical | 0.095 | **0.154** | 0.069 |

On the actual linking task the picture is **mixed**: canonical roughly **doubles
precision (0.073 → 0.154)** but collapses recall — it links fewer things, more
correctly. Precision-favouring is the *right* direction for anti-duplication, but F1
dips, and this is on very thin, noisy pseudo-gold (only 29 exact-subject repeats over
19 protocols), so the linking numbers are not conclusive.

## Two honest caveats

1. **Circularity**: `subject` is produced by the *same* LLM pass as the canonical
   text, so canonical embeddings have a built-in advantage on any subject-derived
   metric. The AUC gain is therefore an **optimistic** estimate.
2. **Scale/label noise**: 19 protocols, exact-subject-match recurrence. Near-duplicate
   subjects that are really the same matter are counted as different, undercounting
   recurrence and destabilising the linking F1.

## The clean, definitive test (recommended next)

Run Tomer's **canonicalization prompt/model on OUR 523 gold segments** (Step 7's real
gold, with real recurrence chains), then re-run Step 10 linking raw-vs-canonical. That
removes both caveats: real gold labels (independent of the canonicalizer) and Step 10's
full 36-repeat benchmark. It is the direct answer to "does canonicalization improve our
linking," and needs one Gemma GPU pass over 523 short segments.

## Files
- `code/build_dataset.py` → `outputs/canon_dataset.json`
- `code/embed_variants.py` (GPU) → `outputs/emb_{raw,canonical}.npy`
- `code/eval_canonical.py` → `outputs/canonical_eval_results.json`, `outputs/fig_canonical.png`
- `sbatch/11_canonicalization_eval.sh`
