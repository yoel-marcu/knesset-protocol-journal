# Step 04 — Hybrid TF-IDF + ABTT Dense Score Fusion

## Method

Score fusion on precomputed 105×105 cosine similarity matrices:

    S_hybrid = α · S_tfidf + (1-α) · S_dense

Sweep α ∈ {0.0, 0.1, ..., 1.0} for each dense model's best-k ABTT embedding
(from Step 03). Clustering: agglomerative (precomputed distance = 1−S) and
spectral (precomputed affinity = max(S, 0)).

## How to Run

```bash
source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT
python steps/04_hybrid_tfidf_dense/code/hybrid_eval.py
```

No GPU required. Runtime: ~2 minutes.

## Key Results

| Configuration | ARI_agg | ARI_spec | AUC |
|---|---|---|---|
| TF-IDF only (baseline) | 0.249 | 0.335 | 0.741 |
| alephbert_k2, pure dense (α=0.0) | 0.513 | 0.377 | 0.887 |
| **alephbert_k2, α=0.6** | **0.569** | **0.519** | 0.879 |
| e5_k1, pure dense (α=0.0) | 0.540 | 0.417 | **0.899** |
| e5_k1, α=0.4 | 0.540 | 0.400 | 0.895 |
| e5_k3, α=0.8 | 0.503 | 0.441 | 0.826 |

**Global best clustering:** alephbert_k2 α=0.6, ARI_agg=0.569 (2.3× TF-IDF baseline).

**Best AUC (Task 2 relevance):** e5_k1 pure dense, AUC=0.899 (+21pp over TF-IDF).

## Findings

1. **Hybrid consistently beats both components alone** for alephbert: α=0.6 (slightly
   more TF-IDF) pushes ARI_agg from 0.513 (pure dense) to 0.569.

2. **e5_k1 is robust to α**: flat curve α=0.0–0.4 (ARI_agg ≈ 0.540), meaning e5_k1
   already captures the lexical signal. Adding TF-IDF neither helps nor hurts much.

3. **mpnet does not benefit** from hybridization; best ARI_agg=0.351 regardless of α.
   Drops from further investigation.

4. **AUC is dominated by pure dense**: e5_k1 AUC=0.899 is the ceiling. Mixing in TF-IDF
   degrades AUC monotonically (TF-IDF similarity gap is only 0.069 vs dense 0.343).

## Recommendation for Task 2 (Subject Linking)

- **Similarity threshold baseline:** use e5_k1 (AUC=0.899) or alephbert_k2 at low α
- **Clustering-based grouping:** use alephbert_k2 @ α=0.6

Best hybrid similarity matrix saved: `outputs/best_sim_matrix.npy` (alephbert_k2, α=0.6).
