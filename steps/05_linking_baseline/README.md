# Step 05 — Streaming Subject Linking: Similarity Threshold Baseline (Task 2a)

## Task

For each new protocol span arriving chronologically: classify as LINK (same subject as
an existing journal entry) or NEW (start a fresh entry). Uses cosine similarity + threshold.
No LLM involved.

**Asymmetric cost:** false-merge (FP, two different subjects merged) >> false-split (FN).
Report both F1-optimal and precision-priority (FMR ≤ 5%) operating points.

## How to Run

```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT
python steps/05_linking_baseline/code/linking_baseline.py
```

No GPU required. ~1 minute.

## Evaluation Modes

**Pairwise (upper bound):** all ordered pairs in recurring subset (105 spans, 28 topics).
Assumes oracle access to all past spans. Gives the ceiling on what similarity alone can do.

**Streaming (realistic):** all 198 spans in chronological order. Journal entries represented
by their first-seen span. Spans not yet seen → journal grows. Errors propagate.

## Results

### Pairwise (oracle)

| Representation | AUC | Best F1 | θ | F1 (FMR≤5%) | θ |
|---|---|---|---|---|---|
| **e5_k1** | **0.899** | **0.531** | 0.33 | **0.503** | 0.26 |
| alephbert_k2 | 0.887 | 0.472 | 0.32 | 0.464 | 0.30 |
| hybrid α=0.6 | 0.879 | 0.463 | 0.20 | 0.463 | 0.20 |

### Streaming (realistic)

| Representation | Best F1 | θ | P | R | F1 (FMR≤5%) | θ |
|---|---|---|---|---|---|---|
| **e5_k1** | **0.411** | 0.52 | 0.419 | 0.403 | 0.092 | 0.79 |
| alephbert_k2 | 0.342 | 0.62 | 0.500 | 0.260 | 0.114 | 0.72 |
| **hybrid α=0.6** | 0.377 | 0.36 | 0.426 | 0.338 | **0.196** | 0.47 |

## Key Findings

1. **Pairwise discrimination is strong (AUC=0.90)** but doesn't directly translate to
   high F1 — the cosine distributions of same/different topics still overlap enough that
   no single threshold achieves high P+R simultaneously.

2. **Streaming gap is real**: pairwise F1=0.531 → streaming F1=0.411 for e5_k1. The
   first-span representative is a noisy proxy for the topic; a centroid-based approach
   would likely close part of this gap.

3. **Anti-duplication operating point is harsh**: at FMR≤5%, streaming recall collapses
   (F1=0.092–0.196). This confirms that similarity thresholds alone cannot satisfy the
   anti-duplication constraint at usable recall — an LLM verifier is needed.

4. **Hybrid α=0.6 has the best FMR≤5% streaming F1 (0.196)**: TF-IDF's lexical precision
   (bill IDs, law names) contributes to avoiding false merges between topically-adjacent
   but legislatively-distinct matters.

## Outputs

| File | Description |
|---|---|
| `outputs/linking_results.json` | Full threshold curves for all 3 representations × 2 modes |
| `outputs/pr_curves_pairwise.png` | P-R curves + false-merge rate vs θ (pairwise) |
| `outputs/pr_curves_streaming.png` | P-R curves + false-merge rate vs θ (streaming) |

## Recommended θ for Downstream Use

| Use case | Representation | θ |
|---|---|---|
| Recall-maximizing (high recall, some false merges OK) | e5_k1 | 0.33 |
| Precision-priority (FMR≤5%, use as retrieval pre-filter) | hybrid α=0.6 | 0.47 |
| LLM verifier input (retrieve top-K candidates) | e5_k1 | — (rank, don't threshold) |
